/**
 * Stage 4 long-run numerical configuration.
 *
 * The HEVI epsilon follows the common forward-centering convention:
 * theta = 0.5 * (1 + epsilon). epsilon=0 is centered Crank-Nicolson;
 * positive epsilon adds selective damping of the vertically propagating
 * acoustic computational mode while leaving the conservative flux form intact.
 */
export const STAGE4_HEVI_OFFCENTERING = 0.1;
