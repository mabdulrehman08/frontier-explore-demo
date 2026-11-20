"""Deterministic planar LiDAR exploration with a 3D point-cloud view."""
import argparse
from collections import deque
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def environment():
    world = np.zeros((48, 64), dtype=bool)
    world[[0, -1], :] = True
    world[:, [0, -1]] = True
    for y, x, h, w in [(8, 14, 14, 6), (30, 14, 10, 6),
                        (8, 34, 8, 15), (26, 34, 14, 6), (26, 50, 8, 5)]:
        world[y:y+h, x:x+w] = True
    return world


def scan(world, known, pose, hits, radius=15):
    # First occupied return terminates each beam; traversed cells are free.
    angles = np.linspace(0, 2*np.pi, 720, endpoint=False)
    distances = np.arange(0, radius, 0.2)
    rays = np.floor(np.array(pose)[None, None, :] + 0.5 +
                    distances[None, :, None] *
                    np.stack([np.sin(angles), np.cos(angles)], axis=1)[:, None, :]).astype(int)
    for ray in rays:
        valid = ((ray[:, 0] >= 0) & (ray[:, 0] < world.shape[0]) &
                 (ray[:, 1] >= 0) & (ray[:, 1] < world.shape[1]))
        ray = ray[:np.argmax(~valid)] if not valid.all() else ray
        if not len(ray):
            continue
        occupied = world[ray[:, 0], ray[:, 1]]
        if occupied.any():
            ray = ray[:np.argmax(occupied) + 1]
            hits.add(tuple(ray[-1]))
        known[ray[:, 0], ray[:, 1]] = world[ray[:, 0], ray[:, 1]].astype(np.int8)


def neighbors(cell, shape):
    y, x = cell
    for yy, xx in [(y-1,x), (y+1,x), (y,x-1), (y,x+1)]:
        if 0 <= yy < shape[0] and 0 <= xx < shape[1]:
            yield yy, xx


def next_path(known, pose):
    # BFS supplies collision-free routes and distance for greedy utility.
    parent = {pose: None}
    queue = deque([pose])
    distance = {pose: 0}
    frontier = []
    while queue:
        cell = queue.popleft()
        adjacent = list(neighbors(cell, known.shape))
        if any(known[n] == -1 for n in adjacent):
            frontier.append(cell)
        for n in adjacent:
            if known[n] == 0 and n not in parent:
                parent[n] = cell
                distance[n] = distance[cell] + 1
                queue.append(n)
    if not frontier:
        return [], frontier
    def utility(c):
        y, x = c
        gain = np.count_nonzero(known[max(0,y-6):y+7, max(0,x-6):x+7] == -1)
        return gain / (distance[c] + 8)
    target = max(frontier, key=utility)
    route = [target]
    while parent[route[-1]] is not None:
        route.append(parent[route[-1]])
    return route[::-1][1:], frontier


def render(known, hits, path, frontier, coverage, step, out):
    fig = plt.figure(figsize=(12, 5), facecolor="#101827")
    ax = fig.add_subplot(121)
    ax.imshow(known, origin="lower", vmin=-1, vmax=1,
              cmap=ListedColormap(["#253247", "#b9e4dc", "#f0b35a"]))
    if frontier:
        fy, fx = np.array(frontier).T
        ax.scatter(fx, fy, s=4, c="#ef628e", label="Frontiers")
    py, px = np.array(path).T
    ax.plot(px, py, color="#397be8", linewidth=2, label="Drone path")
    ax.scatter(px[-1], py[-1], c="red", s=35)
    ax.set_title(f"Scan {step:02d} | free-space coverage {coverage:.1%}")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("x (0.5 m voxels)")
    ax.set_ylabel("y (0.5 m voxels)")
    view = fig.add_subplot(122, projection="3d")
    if hits:
        points = np.array(sorted(hits))
        # Extrude planar returns into a synthetic vertical wall cloud.
        xyz = np.array([(x*.5, y*.5, z) for y, x in points
                        for z in np.linspace(0, 3, 7)])
        view.scatter(*xyz.T, c=xyz[:, 2], cmap="viridis", s=1, alpha=.6)
    view.plot(px*.5, py*.5, np.full(len(px), 1.5), c="#397be8")
    view.set(xlim=(0,32), ylim=(0,24), zlim=(0,4),
             xlabel="x / m", ylabel="y / m", zlabel="z / m",
             title="Accumulated synthetic LiDAR returns")
    view.view_init(55, -65)
    fig.tight_layout()
    fig.savefig(out / f"frame_{step:03d}.png", dpi=110)
    if step == 0:
        fig.savefig(out / "start.png", dpi=110)
    fig.savefig(out / "example.png", dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--out", type=Path, default=Path("out"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    world = environment()
    known = np.full(world.shape, -1, dtype=np.int8)
    pose = (4, 4)
    path, hits, history = [pose], set(), []
    for step in range(args.steps):
        scan(world, known, pose, hits)
        route, frontier = next_path(known, pose)
        coverage = np.count_nonzero((known == 0) & ~world) / np.count_nonzero(~world)
        history.append(coverage)
        render(known, hits, path, frontier, coverage, step, args.out)
        if not route:
            break
        # Limit motion between scans; every segment stays in known free space.
        segment = route[:8]
        assert all(known[c] == 0 and not world[c] for c in segment)
        path.extend(segment)
        pose = path[-1]
    assert np.all(np.diff(history) >= 0), "Map coverage must be monotonic"
    np.savez_compressed(args.out / "map.npz", occupancy=known,
                        path=np.array(path), coverage=history)
    print(f"{len(history)} scans; {history[-1]:.1%} free-space coverage; {len(hits)} wall returns")


if __name__ == "__main__":
    main()
