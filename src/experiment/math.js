export function createRng(seed = 0) {
  let state = (seed >>> 0) || 1;
  let spare = null;

  function uniform() {
    state += 0x6d2b79f5;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  function int(maxExclusive) {
    return Math.floor(uniform() * maxExclusive);
  }

  function normal() {
    if (spare !== null) {
      const z = spare;
      spare = null;
      return z;
    }

    let u = 0;
    let v = 0;
    while (u === 0) {
      u = uniform();
    }
    while (v === 0) {
      v = uniform();
    }

    const mag = Math.sqrt(-2.0 * Math.log(u));
    const theta = 2.0 * Math.PI * v;
    spare = mag * Math.sin(theta);
    return mag * Math.cos(theta);
  }

  return { uniform, int, normal };
}

export function zeros(rows, cols) {
  const out = new Array(rows);
  for (let i = 0; i < rows; i += 1) {
    out[i] = new Array(cols).fill(0);
  }
  return out;
}

export function cloneMatrix(mat) {
  return mat.map((row) => row.slice());
}

export function dotVec(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i += 1) {
    s += a[i] * b[i];
  }
  return s;
}

export function l2Norm(arr) {
  let s = 0;
  for (let i = 0; i < arr.length; i += 1) {
    s += arr[i] * arr[i];
  }
  return Math.sqrt(s);
}

export function flattenGradients(gradA, gradB) {
  const total = gradA.length * gradA[0].length + gradB.length * gradB[0].length;
  const out = new Array(total);
  let k = 0;

  for (let i = 0; i < gradA.length; i += 1) {
    for (let j = 0; j < gradA[i].length; j += 1) {
      out[k] = gradA[i][j];
      k += 1;
    }
  }
  for (let i = 0; i < gradB.length; i += 1) {
    for (let j = 0; j < gradB[i].length; j += 1) {
      out[k] = gradB[i][j];
      k += 1;
    }
  }

  return out;
}

export function unflattenGradients(flat, rank, dIn, nAssets) {
  const gradA = zeros(rank, dIn);
  const gradB = zeros(nAssets, rank);
  let k = 0;

  for (let i = 0; i < rank; i += 1) {
    for (let j = 0; j < dIn; j += 1) {
      gradA[i][j] = flat[k];
      k += 1;
    }
  }

  for (let i = 0; i < nAssets; i += 1) {
    for (let j = 0; j < rank; j += 1) {
      gradB[i][j] = flat[k];
      k += 1;
    }
  }

  return { gradA, gradB };
}

export function projectAgem(gNew, gMem, eps = 0) {
  let dot = 0;
  let denom = 0;

  for (let i = 0; i < gNew.length; i += 1) {
    dot += gNew[i] * gMem[i];
    denom += gMem[i] * gMem[i];
  }

  if (dot >= -eps) {
    return { projected: gNew.slice(), isProjected: false, dot };
  }

  const scale = (dot + eps) / (denom + 1e-12);
  const projected = new Array(gNew.length);
  for (let i = 0; i < gNew.length; i += 1) {
    projected[i] = gNew[i] - scale * gMem[i];
  }

  return { projected, isProjected: true, dot };
}
