find ~/data/ztf/ztfcosmoidr/dr2/lightcurves/ -iname *.csv -exec python3 estimate_lc_params.py {} 50 100 2 False ~/data/ztf/lc/ \;
