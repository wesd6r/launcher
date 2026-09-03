import { describe, expect, it, vi } from 'vitest';

// The util module imports `app` and `screen` from electron. In the test environment the electron binary is not
// available, so mock the module before importing the module under test.
vi.mock('electron', () => ({
  app: { getPath: vi.fn(() => '') },
  screen: {},
}));

import { getTorchPlatform } from './util';

describe('getTorchPlatform', () => {
  it('maps the intel GPU to xpu', () => {
    expect(getTorchPlatform('intel')).toBe('xpu');
  });

  it('maps the amd GPU to rocm', () => {
    expect(getTorchPlatform('amd')).toBe('rocm');
  });

  it('maps nvidia GPUs to cuda', () => {
    expect(getTorchPlatform('nvidia<30xx')).toBe('cuda');
    expect(getTorchPlatform('nvidia>=30xx')).toBe('cuda');
  });

  it('maps nogpu to cpu', () => {
    expect(getTorchPlatform('nogpu')).toBe('cpu');
  });
});
