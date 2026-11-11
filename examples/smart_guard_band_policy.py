from __future__ import annotations

from typing import Dict, Optional, Set
import numpy as np
from spectrum.policy import Observation, Policy


class SmartGuardBandPolicy(Policy):
    """O-RAN Spektrum Paylaşımı için Gelişmiş PRB Koruma Politikası.

    Özellikler:
    1. EWMA Filtreleme: Gürültü sıçramalarını sönümlemek için üstel hareketli ortalama.
    2. Dinamik Medyan Eşikleme: Ortam gürültü tabanına göre otomatik SNR hesaplama.
    3. Koruma Bandı (Guard-Band): Şiddetli parazit (>severe_margin_db) görülen PRB'lerin
       yan lob sızıntılarını önlemek için komşu PRB'leri otomatik bloklama.
    4. Histerezis & Hold Time: Sinyal kesilse dahi ayarlanabilir süre boyunca bloğu koruma.

    Kullanım:
        python3 -m examples.spectrum_dapp --link zmq --transport ipc --encoding-method asn1 \\
            --control -ru rfsim --policy examples.smart_guard_band_policy:SmartGuardBandPolicy
    """

    def __init__(
        self,
        margin_db: float = 9.0,
        severe_margin_db: float = 17.0,
        alpha: float = 0.45,
        hold_s: float = 0.1,
        guard_prbs: int = 1,
        history_len: int = 64,
    ) -> None:
        super().__init__(history_len=history_len)
        self.margin_db = margin_db
        self.severe_margin_db = severe_margin_db
        self.alpha = alpha
        self.hold_s = hold_s
        self.guard_prbs = guard_prbs
        self._ema_power: Optional[np.ndarray] = None
        self._last_hot: Dict[int, float] = {}

    def reset(self) -> None:
        super().reset()
        self._ema_power = None
        self._last_hot.clear()

    def update(self, obs: Observation) -> Set[int]:
        level = np.where(np.isnan(obs.prb_power_db), 0.0, obs.prb_power_db)
        if self._ema_power is None:
            self._ema_power = level.copy()
        else:
            self._ema_power = self.alpha * level + (1.0 - self.alpha) * self._ema_power

        # Medyan gürültü tabanı tahmini
        floor = float(np.median(self._ema_power))
        snr = self._ema_power - floor

        # Normal parazit eşiği
        hot_prbs = set(int(p) for p in np.where(snr > self.margin_db)[0])
        # Şiddetli parazit eşiği
        severe_prbs = [int(p) for p in np.where(snr > self.severe_margin_db)[0]]

        # Şiddetli parazit olan PRB'lerin komşularına guard-band koruması uygula
        for p in severe_prbs:
            for g in range(1, self.guard_prbs + 1):
                if p - g >= 0:
                    hot_prbs.add(p - g)
                if p + g < obs.num_prbs:
                    hot_prbs.add(p + g)

        # Son görülme zamanını kaydet
        for p in hot_prbs:
            self._last_hot[p] = obs.t

        # Tutma süresi (hold time) dolan PRB'leri serbest bırak
        self._last_hot = {p: t for p, t in self._last_hot.items() if (obs.t - t) <= self.hold_s}
        return set(self._last_hot)

