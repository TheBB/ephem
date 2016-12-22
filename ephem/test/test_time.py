import pytest
import novas.compat as nv

from ephem.time import angle, datetime


@pytest.fixture
def now():
    return datetime.utcnow()


def test_era(now):
    my_era = now.era

    hi, lo = divmod(now.ut1.jd, 1)
    nv_era = angle.from_degrees(nv.era(hi, lo))

    assert my_era == pytest.approx(nv_era)


def test_gmst(now):
    my_gmst = now.gmst

    hi, lo = divmod(now.ut1.jd, 1)
    dt = now.tt - now.ut1
    dt = dt.seconds + dt.microseconds * 1e-6
    nv_gmst = angle.from_hours(nv.sidereal_time(hi, lo, dt, gst_type=0))

    assert my_gmst == pytest.approx(nv_gmst)
