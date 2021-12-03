#find ~/data/ztf/ztfcosmoidr/dr2/lightcurves/ -iname *.csv -print0 | parallel -0 python3 plot_lc.py {} 0 ~/data/ztf/lc/
find ~/data/ztf/ztfcosmoidr/dr2/lightcurves/ -iname *.csv -print0 | parallel -0 python3 lightcurve_control.py {} 50 100 2 False ~/data/ztf/lc2/
