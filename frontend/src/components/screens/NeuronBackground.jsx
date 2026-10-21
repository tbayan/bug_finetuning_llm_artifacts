import { useRef, useEffect } from 'react';

const NODE_COUNT = 32;
const CONNECT_DISTANCE = 160;
const NODE_RADIUS = 6;
const NODE_PULSE_RANGE = 2;
const LINE_WIDTH = 1.5;
const LINE_OPACITY_MAX = 0.55;
const NODE_COLOR = 'rgba(179, 227, 253, 0.95)';
const DRIFT = 0.4;
const PULSE_SPEED = 0.003;

function rand(lo, hi) {
  return lo + Math.random() * (hi - lo);
}

export default function NeuronBackground() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    let width = 0;
    let height = 0;
    let nodes = [];
    let animationId = 0;
    let time = 0;

    const setSize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const rect = canvas.getBoundingClientRect();
      width = rect.width;
      height = rect.height;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      ctx.scale(dpr, dpr);
      initNodes();
    };

    const initNodes = () => {
      nodes = [];
      for (let i = 0; i < NODE_COUNT; i++) {
        nodes.push({
          x: rand(0.05 * width, 0.95 * width),
          y: rand(0.05 * height, 0.95 * height),
          vx: rand(-DRIFT, DRIFT),
          vy: rand(-DRIFT, DRIFT),
          pulse: Math.random() * Math.PI * 2,
        });
      }
    };

    const tick = () => {
      time += 1;
      ctx.clearRect(0, 0, width, height);

      for (const n of nodes) {
        n.x += n.vx;
        n.y += n.vy;
        n.pulse += PULSE_SPEED;
        if (n.x <= 0 || n.x >= width) n.vx *= -1;
        if (n.y <= 0 || n.y >= height) n.vy *= -1;
        n.x = Math.max(0, Math.min(width, n.x));
        n.y = Math.max(0, Math.min(height, n.y));
      }

      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const a = nodes[i];
          const b = nodes[j];
          const dx = b.x - a.x;
          const dy = b.y - a.y;
          const d = Math.hypot(dx, dy);
          if (d < CONNECT_DISTANCE) {
            const opacity = LINE_OPACITY_MAX * (1 - d / CONNECT_DISTANCE);
            ctx.strokeStyle = `rgba(179, 227, 253, ${opacity})`;
            ctx.lineWidth = LINE_WIDTH;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.stroke();
          }
        }
      }

      for (const n of nodes) {
        const r = NODE_RADIUS + NODE_PULSE_RANGE * Math.sin(n.pulse);
        ctx.fillStyle = NODE_COLOR;
        ctx.beginPath();
        ctx.arc(n.x, n.y, Math.max(3, r), 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = 'rgba(179, 227, 253, 0.6)';
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      animationId = requestAnimationFrame(tick);
    };

    setSize();
    tick();

    const onResize = () => {
      setSize();
    };
    window.addEventListener('resize', onResize);

    return () => {
      window.removeEventListener('resize', onResize);
      cancelAnimationFrame(animationId);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 h-full w-full"
      aria-hidden
      style={{ display: 'block' }}
    />
  );
}
