import { useEffect, useRef } from 'react';

interface SpinningCoinProps {
  size?: number;
  className?: string;
}

export function SpinningCoin({ size = 400, className = '' }: SpinningCoinProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Configuration
    const COIN_COLOR = '#9EF6CA';

    const pixelWidth = Math.floor(size / 40);
    const pixelHeight = Math.floor(size / 40);

    const screenWidth = Math.floor(size / pixelWidth);
    const screenHeight = Math.floor(size / pixelHeight);
    const screenSize = screenWidth * screenHeight;

    // Rotation angles
    let A = 0.0;
    let B = 0.0;

    // Rotation speeds
    const SPIN_A = 0.02;
    const SPIN_B = 0.0075;

    // Sampling density
    const angleSpacing = 6;
    const radialSpacing = 1;
    const heightSpacing = 1;

    // ASCII luminance characters
    const faceChars = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";
    const rimChars = "||||/\\IIII";

    // Projection constants
    const K2 = 200;
    const K1 = (screenHeight * K2 * 3) / (8 * 30);

    // Coin geometry
    const COIN_RADIUS = 26;
    const COIN_HALF_THICK = 2.5;

    // Set font
    const fontSize = Math.max(8, Math.floor(pixelHeight * 0.9));
    ctx.font = `bold ${fontSize}px 'Courier New', monospace`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    // Helper functions
    function rotateXZ(
      vx: number,
      vy: number,
      vz: number,
      A: number,
      B: number
    ): [number, number, number] {
      // Rotate around X, then around Z
      const y1 = vy * Math.cos(A) - vz * Math.sin(A);
      const z1 = vy * Math.sin(A) + vz * Math.cos(A);
      const x1 = vx;
      const x2 = x1 * Math.cos(B) - y1 * Math.sin(B);
      const y2 = x1 * Math.sin(B) + y1 * Math.cos(B);
      return [x2, y2, z1];
    }

    function dot(
      ax: number,
      ay: number,
      az: number,
      bx: number,
      by: number,
      bz: number
    ): number {
      return ax * bx + ay * by + az * bz;
    }

    function normalize(x: number, y: number, z: number): [number, number, number] {
      const m = Math.sqrt(x * x + y * y + z * z) || 1.0;
      return [x / m, y / m, z / m];
    }

    function atan2Approx(y: number, x: number): number {
      if (x === 0) {
        return y > 0 ? 1.5708 : -1.5708;
      }
      const a = y / x;
      if (x > 0) {
        return Math.abs(a) <= 1
          ? a / (1 + 0.28 * a * a)
          : (1.5708 - a / (a * a + 0.28)) * (a > 0 ? 1 : -1);
      } else {
        return Math.abs(a) <= 1
          ? a / (1 + 0.28 * a * a) + 3.14159
          : (1.5708 - a / (a * a + 0.28)) * (a > 0 ? 1 : -1) + 3.14159;
      }
    }

    function getCoinDesignChar(xLocal: number, zLocal: number, r: number): string | null {
      const distFromCenter = Math.sqrt(xLocal ** 2 + zLocal ** 2);

      // Outer raised ring
      if (COIN_RADIUS - 3 < distFromCenter && distFromCenter <= COIN_RADIUS - 0.5) {
        return '@';
      }

      // Text ring
      if (16 < distFromCenter && distFromCenter < 19) {
        const angle = Math.floor(((atan2Approx(zLocal, xLocal) + 3.14159) * 5) % 10);
        if (angle % 2 === 0) {
          return '$';
        }
        return '@';
      }

      // Middle decorative ring
      if (11 < distFromCenter && distFromCenter < 13) {
        return '#';
      }

      // Center star/emblem
      if (distFromCenter < 6) {
        if (Math.abs(xLocal) < 1.5 || Math.abs(zLocal) < 1.5) {
          return '#';
        }
        if (Math.abs(Math.abs(xLocal) - Math.abs(zLocal)) < 1) {
          return '#';
        }
      }

      return null;
    }

    // Light direction
    const lightDir = normalize(0.3, 1.0, -0.8);

    // Animation loop
    function animate() {
      // Clear canvas to transparent
      ctx.clearRect(0, 0, size, size);

      const output: string[] = new Array(screenSize).fill(' ');
      const zbuffer: number[] = new Array(screenSize).fill(0.0);

      // Render rim (cylindrical surface)
      for (let phi = 0; phi < 628; phi += angleSpacing) {
        const cph = Math.cos(phi / 100.0);
        const sph = Math.sin(phi / 100.0);
        const nxLocal = cph;
        const nyLocal = 0.0;
        const nzLocal = sph;

        for (
          let h = -Math.floor(COIN_HALF_THICK * 10);
          h <= Math.floor(COIN_HALF_THICK * 10);
          h += heightSpacing * 10
        ) {
          const hVal = h / 10.0;
          const xLocal = COIN_RADIUS * cph;
          const yLocal = hVal;
          const zLocal = COIN_RADIUS * sph;

          const [x, y, z] = rotateXZ(xLocal, yLocal, zLocal, A, B);
          const zProj = z + K2;
          if (zProj <= 0) continue;

          const ooz = 1.0 / zProj;
          const xp = Math.floor(screenWidth / 2 + K1 * ooz * x);
          const yp = Math.floor(screenHeight / 2 - K1 * ooz * y);
          if (!(xp >= 0 && xp < screenWidth && yp >= 0 && yp < screenHeight)) continue;

          const [nx, ny, nz] = rotateXZ(nxLocal, nyLocal, nzLocal, A, B);
          const [nxNorm, nyNorm, nzNorm] = normalize(nx, ny, nz);

          let L = dot(nxNorm, nyNorm, nzNorm, ...lightDir);
          L = Math.max(0, L) ** 1.2;
          L = L * 0.8 + 0.2; // Add ambient light

          if (L <= 0) continue;

          const pos = xp + screenWidth * yp;
          if (ooz > zbuffer[pos]) {
            zbuffer[pos] = ooz;
            const luminanceIndex = Math.floor(L * (rimChars.length - 1));
            output[pos] = rimChars[Math.max(0, Math.min(luminanceIndex, rimChars.length - 1))];
          }
        }
      }

      // Render top & bottom faces
      for (const faceSign of [1.0, -1.0]) {
        const nyLocal = faceSign;
        for (let r = 0; r <= Math.floor(COIN_RADIUS * 10); r += radialSpacing * 10) {
          const rVal = r / 10.0;
          for (let phi = 0; phi < 628; phi += angleSpacing) {
            const cph = Math.cos(phi / 100.0);
            const sph = Math.sin(phi / 100.0);

            const xLocal = rVal * cph;
            const yLocal = faceSign * COIN_HALF_THICK;
            const zLocal = rVal * sph;

            const [x, y, z] = rotateXZ(xLocal, yLocal, zLocal, A, B);
            const zProj = z + K2;
            if (zProj <= 0) continue;

            const ooz = 1.0 / zProj;
            const xp = Math.floor(screenWidth / 2 + K1 * ooz * x);
            const yp = Math.floor(screenHeight / 2 - K1 * ooz * y);
            if (!(xp >= 0 && xp < screenWidth && yp >= 0 && yp < screenHeight)) continue;

            const [nx, ny, nz] = rotateXZ(0.0, nyLocal, 0.0, A, B);
            const [nxNorm, nyNorm, nzNorm] = normalize(nx, ny, nz);

            let L = dot(nxNorm, nyNorm, nzNorm, ...lightDir);
            L = Math.max(0, L);
            const specular = Math.max(0, L) ** 3;
            L = L ** 1.2 * 0.7 + specular * 0.3;
            L = L * 0.85 + 0.15; // Add ambient light

            if (L <= 0.15) continue;

            const pos = xp + screenWidth * yp;
            if (ooz > zbuffer[pos]) {
              zbuffer[pos] = ooz;

              // Check for coin design pattern
              const designChar = getCoinDesignChar(xLocal, zLocal, rVal);

              if (designChar) {
                output[pos] = designChar;
              } else {
                const luminanceIndex = Math.floor(L * (faceChars.length - 1));
                output[pos] =
                  faceChars[Math.max(0, Math.min(luminanceIndex, faceChars.length - 1))];
              }
            }
          }
        }
      }

      // Draw ASCII characters
      ctx.fillStyle = COIN_COLOR;
      let k = 0;
      for (let i = 0; i < screenHeight; i++) {
        for (let j = 0; j < screenWidth; j++) {
          const xPixel = j * pixelWidth + pixelWidth / 2;
          const yPixel = i * pixelHeight + pixelHeight / 2;
          ctx.fillText(output[k], xPixel, yPixel);
          k++;
        }
      }

      // Update rotation
      A += SPIN_A;
      B += SPIN_B;

      animationRef.current = requestAnimationFrame(animate);
    }

    // Start animation
    animate();

    // Cleanup
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [size]);

  return (
    <canvas
      ref={canvasRef}
      width={size}
      height={size}
      className={className}
      style={{ imageRendering: 'pixelated' }}
    />
  );
}
