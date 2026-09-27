// Mouse / touch orbit controls shared by the globe and the regional volume view:
// one pointer drags to rotate, two pointers pinch to zoom, the wheel zooms, a short tap picks.

export interface OrbitHandlers {
  rotate(dx: number, dy: number): void;      // pixels moved
  zoom(factor: number): void;                // > 1 zooms out (larger distance)
  tap?(clientX: number, clientY: number): void;
}

export function attachOrbit(canvas: HTMLCanvasElement, h: OrbitHandlers): void {
  const pts = new Map<number, { x: number; y: number }>();
  let moved = 0, pinch0 = 0, multi = false;
  const spread = (): number => { const p = [...pts.values()]; return p.length < 2 ? 0 : Math.hypot(p[0]!.x - p[1]!.x, p[0]!.y - p[1]!.y); };
  canvas.addEventListener('pointerdown', (e) => {
    canvas.setPointerCapture(e.pointerId);
    pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pts.size === 1) { moved = 0; multi = false; }
    if (pts.size === 2) { pinch0 = spread(); multi = true; }
  });
  const end = (e: PointerEvent): void => {
    const was = pts.get(e.pointerId);
    pts.delete(e.pointerId);
    if (was && pts.size === 0 && !multi && moved < 6 && h.tap) h.tap(e.clientX, e.clientY);
    if (pts.size < 2) pinch0 = 0;
  };
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointercancel', end);
  canvas.addEventListener('pointermove', (e) => {
    const p = pts.get(e.pointerId);
    if (!p) return;
    const dx = e.clientX - p.x, dy = e.clientY - p.y;
    p.x = e.clientX; p.y = e.clientY;
    if (pts.size >= 2) {
      const s = spread();
      if (pinch0 > 0 && s > 0) { h.zoom(pinch0 / s); pinch0 = s; }
      return;
    }
    moved += Math.abs(dx) + Math.abs(dy);
    h.rotate(dx, dy);
  });
  canvas.addEventListener('wheel', (e) => { e.preventDefault(); h.zoom(Math.exp(e.deltaY * 0.001)); }, { passive: false });
}
