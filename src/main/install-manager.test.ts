import { describe, expect, it, vi } from 'vitest';

// The install-manager module imports `ipcMain` from electron. In the test environment the electron binary is not
// available, so mock the module before importing the module under test.
vi.mock('electron', () => ({
  ipcMain: { handle: vi.fn(), removeHandler: vi.fn() },
}));

import { getInvokeExtras } from './install-manager';

describe('getInvokeExtras', () => {
  it('maps the intel GPU to the xpu extra on non-darwin platforms', () => {
    expect(getInvokeExtras('intel', 'xpu')).toEqual(['xpu']);
  });

  it('returns an empty extra list on the legacy path when torchPlatform is null', () => {
    // The legacy install path passes null as torchPlatform, so no torch extra is requested. This is exactly
    // why XPU support must be rejected earlier: with invokeExtras = [] the install silently resolves torch
    // from the default PyPI index instead of the xpu index.
    expect(getInvokeExtras('intel', null)).toEqual([]);
  });

  it('includes xformers only for 20xx and earlier Nvidia GPUs', () => {
    expect(getInvokeExtras('nvidia<30xx', 'cuda')).toEqual(['cuda', 'xformers']);
    expect(getInvokeExtras('nvidia>=30xx', 'cuda')).toEqual(['cuda']);
  });

  it('includes the rocm extra for amd', () => {
    expect(getInvokeExtras('amd', 'rocm')).toEqual(['rocm']);
  });

  it('adds no torch extra for nogpu', () => {
    expect(getInvokeExtras('nogpu', 'cpu')).toEqual(['cpu']);
  });
});
