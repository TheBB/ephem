import click
from math import pi

from .time import angle, datetime


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx):
    if ctx.invoked_subcommand is None:
        ctx.invoke(time)


@main.command()
def time():
    t = datetime.utcnow()
    print('      UTC:', t)
    print('      UT1:', t.ut1)
    print('      TAI:', t.tai)
    print('       TT:', t.tt)
    print('  JD(UT1):', t.jd)
    print(' MJD(UT1):', t.mjd)
    print('      ERA:', t.era.time)
    print('     GMST:', t.gmst.time)


if __name__ == '__main__':
    main()
