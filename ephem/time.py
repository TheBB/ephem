from collections import namedtuple
import datetime as dt
from functools import lru_cache
from math import ceil, cos, floor, pi, sin
from numbers import Number
import requests

tau = 2 * pi

LeapSecond = namedtuple('LeapSecond', ['time', 'diff'])

LEAP_SECONDS = [
    LeapSecond(dt.datetime(1972,  6, 30, 23, 59, 59),  1),
    LeapSecond(dt.datetime(1972, 12, 31, 23, 59, 59),  1),
    LeapSecond(dt.datetime(1973, 12, 31, 23, 59, 59),  1),
    LeapSecond(dt.datetime(1974, 12, 31, 23, 59, 59),  1),
    LeapSecond(dt.datetime(1975, 12, 31, 23, 59, 59),  1),
    LeapSecond(dt.datetime(1976, 12, 31, 23, 59, 59),  1),
    LeapSecond(dt.datetime(1977, 12, 31, 23, 59, 59),  1),
    LeapSecond(dt.datetime(1978, 12, 31, 23, 59, 59),  1),
    LeapSecond(dt.datetime(1979, 12, 31, 23, 59, 59),  1),
    LeapSecond(dt.datetime(1981,  6, 30, 23, 59, 59),  1),
    LeapSecond(dt.datetime(1982,  6, 30, 23, 59, 59),  1),
    LeapSecond(dt.datetime(1983,  6, 30, 23, 59, 59),  1),
    LeapSecond(dt.datetime(1985,  6, 30, 23, 59, 59),  1),
    LeapSecond(dt.datetime(1987, 12, 31, 23, 59, 59),  1),
    LeapSecond(dt.datetime(1989, 12, 31, 23, 59, 59),  1),
    LeapSecond(dt.datetime(1990, 12, 31, 23, 59, 59),  1),
    LeapSecond(dt.datetime(1992,  6, 30, 23, 59, 59),  1),
    LeapSecond(dt.datetime(1993,  6, 30, 23, 59, 59),  1),
    LeapSecond(dt.datetime(1994,  6, 30, 23, 59, 59),  1),
    LeapSecond(dt.datetime(1995, 12, 31, 23, 59, 59),  1),
    LeapSecond(dt.datetime(1997,  6, 30, 23, 59, 59),  1),
    LeapSecond(dt.datetime(1998, 12, 31, 23, 59, 59),  1),
    LeapSecond(dt.datetime(2005, 12, 31, 23, 59, 59),  1),
    LeapSecond(dt.datetime(2008, 12, 31, 23, 59, 59),  1),
    LeapSecond(dt.datetime(2012,  6, 30, 23, 59, 59),  1),
    LeapSecond(dt.datetime(2015,  6, 30, 23, 59, 59),  1),
    LeapSecond(dt.datetime(2016, 12, 31, 23, 59, 59),  1),
]


@lru_cache(maxsize=1)
def utdiffs():
    text = requests.get('https://datacenter.iers.org/eop/-/somos/5Rgv/latest/6').text
    diffs = {}
    for line in text.split('\n'):
        args = line.split()

        try:
            _, _, _, mjd = map(int, args[:4])
            _, _, _, _, diff, _ = map(float, args[4:])
            diffs[mjd] = diff
        except ValueError: pass

        try:
            _, _, _, mjd = map(int, args[:4])
            _, _, diff = map(float, args[4:])
            diffs[mjd] = diff
        except ValueError: pass

    return diffs


class datetime(dt.datetime):

    @property
    def ut1(self):
        """Universal time (mean solar time)."""
        diffs = utdiffs()
        mjd = self.mjd
        lo, hi, fr = int(floor(mjd)), int(ceil(mjd)), mjd % 1
        diff = (1 - fr) * diffs[lo] + fr * diffs[hi]
        return self + dt.timedelta(seconds=diff)

    @property
    def ut2(self):
        """Smoothed universal time."""
        bessel = (self.tt.jd - 2415020.31362) / 365.242198781
        bessel = (bessel % 1) * tau
        diff = 0.022 * sin(bessel)
        diff -= 0.012 * cos(bessel)
        diff -= 0.006 * sin(2 * bessel)
        diff += 0.007 * cos(2 * bessel)
        return self.ut1 + dt.timedelta(seconds=diff)

    @property
    def tai(self):
        """International Atomic Time."""
        delta = 10 + sum(ls.diff for ls in LEAP_SECONDS if self > ls.time)
        return self + dt.timedelta(seconds=delta)

    @property
    def tt(self):
        """Terrestrial time."""
        return self.tai + dt.timedelta(seconds=32.184)

    @property
    def jd(self):
        """Julian day number."""
        # Midnight 2000-01-01 was JD 2451544.5
        epoch = dt.datetime(2000, 1, 1, 0, 0, 0)
        jd = (self - epoch).days + 2451544.5
        jd += angle(self) / tau
        return jd

    @property
    def mjd(self):
        """Modified Julian day number."""
        return self.jd - 2400000.5

    @property
    def era(self):
        """Earth rotation angle."""
        ret = self.ut1.jd - 2451545
        ret *= 1.00273781191135448
        ret += 0.7790572732640
        return angle(ret * tau)

    @property
    def _gmst_pol(self):
        """Polynomial part of GMST, to fifth order."""
        T = (self.tt.jd - 2451545) / 36525
        arcsec = pi / 180 / 60 / 60
        ret = 0.014506
        ret += 4612.156534 * T
        ret += 1.3915817 * T**2
        ret -= 0.00000044 * T**3
        ret -= 0.000029956 * T**4
        ret -= 3.86e-8 * T**5
        return angle(ret * arcsec)

    @property
    def gmst(self):
        """Greenwich mean sidereal time."""
        return self.era + self._gmst_pol

    def __add__(self, other):
        r = super(datetime, self).__add__(other)
        if isinstance(r, dt.datetime):
            return datetime(
                r.year, r.month, r.day, r.hour,
                r.minute, r.second, r.microsecond
            )
        return r

    def __sub__(self, other):
        r = super(datetime, self).__sub__(other)
        if isinstance(r, dt.datetime):
            return datetime(
                r.year, r.month, r.day, r.hour,
                r.minute, r.second, r.microsecond
            )
        return r


class angle(float):

    def __new__(cls, arg):
        try:
            return float.__new__(cls, arg % tau)
        except TypeError: pass

        try:
            r = arg.hour / 24
            r += arg.minute / 24 / 60
            r += arg.second / 24 / 60 / 60
            r += arg.microsecond / 24 / 60 / 60 / 1e6
            return float.__new__(cls, (r % 1) * tau)
        except AttributeError: pass

        raise TypeError()

    @classmethod
    def from_degrees(cls, arg):
        return cls(arg / 180 * pi)

    @classmethod
    def from_hours(cls, arg):
        return cls(arg / 24 * tau)

    @property
    def time(self):
        r = self / tau
        hour, r = divmod(24 * r, 1)
        minute, r = divmod(60 * r, 1)
        second, r = divmod(60 * r, 1)
        microsecond, r = divmod(1e6 * r, 1)
        return dt.time(int(hour), int(minute), int(second), int(microsecond))

    @property
    def timedelta(self):
        return dt.timedelta(days=self/tau)

    @property
    def hour(self):
        return self.time.hour

    @property
    def minute(self):
        return self.time.minute

    @property
    def second(self):
        return self.time.second

    @property
    def microsecond(self):
        return self.time.microsecond

    def __add__(self, other):
        return angle(float.__add__(self, other))

    def __sub__(self, other):
        return angle(float.__sub__(self, other))

    def __abs__(self):
        if self > pi:
            return float.__abs__(2 * pi - self)
        return float.__abs__(self)
