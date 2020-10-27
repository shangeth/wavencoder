class SpeedChange:

    def __init__(self, factor_range=(-0.15, 0.15), report=False):
        self.factor_range = factor_range
        self.report = report

    # @profile
    def __call__(self, pkg):
        pkg = format_package(pkg)
        wav = pkg['chunk']
        wav = wav.data.numpy().reshape(-1).astype(np.float32)
        warp_factor = random.random() * (self.factor_range[1] - \
                                         self.factor_range[0]) + \
                      self.factor_range[0]
        samp_warp = wav.shape[0] + int(warp_factor * wav.shape[0])
        rwav = signal.resample(wav, samp_warp)
        if len(rwav) > len(wav):
            mid_i = (len(rwav) // 2) - len(wav) // 2
            rwav = rwav[mid_i:mid_i + len(wav)]
        if len(rwav) < len(wav):
            diff = len(wav) - len(rwav)
            P = (len(wav) - len(rwav)) // 2
            if diff % 2 == 0:
                rwav = np.concatenate((np.zeros(P, ),
                                       wav,
                                       np.zeros(P, )),
                                      axis=0)
            else:
                rwav = np.concatenate((np.zeros(P, ),
                                       wav,
                                       np.zeros(P + 1, )),
                                      axis=0)
        if self.report:
            if 'report' not in pkg:
                pkg['report'] = {}
            pkg['report']['warp_factor'] = warp_factor
        pkg['chunk'] = torch.FloatTensor(rwav)
        return pkg