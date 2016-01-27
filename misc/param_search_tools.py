import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import glob
from itertools import izip, product
from os.path import join, exists
import os
from sklearn import linear_model

import LMR_utils2 as utils2
import LMR_gridded as lmrgrid
import LMR_config

mpl.rcParams['figure.figsize'] = (10., 6.)
mpl.rcParams['font.size'] = 16

def compile_gmts(parent_dir, a_d_vals=None):

    # Directories for each parameter value
    if a_d_vals is not None:
        ad_dir = 'a{:1.1f}_d{:1.2f}'
        param_iters = [join(parent_dir, ad_dir.format(a, d))
                       for a, d in a_d_vals]
    else:
        param_iters = glob.glob(join(parent_dir, 'r*'))

    gmts = None
    times = None

    # Load the analysis GMT timeseries for each parameter value
    for k, f in enumerate(param_iters):
        try:
            npdict = np.load(join(f, 'gmt.npz'))
            if times is None:
                times = npdict['recon_times']
                gmts = np.zeros((len(param_iters), len(times))) * np.nan
            gmts[k] = npdict['gmt_save'][-1]
        except IOError as e:
            print e

    return times, gmts


def center_to_time_range(times, analysis, time_axis, trange=(1900, 2000)):

    start, end = trange

    idx_20c, = np.where((times >= start) & (times <= end))
    mean_20c = analysis.take(idx_20c, axis=time_axis).mean(axis=time_axis,
                                                           keepdims=True)
    centered = analysis - mean_20c

    return centered


def compile_iters(iter_folder, analysis_gmt, analysis_times, a_d_vals,
                  trange=(1880, 2000), center_trange=(1900, 2000)):

    iter_folder_list = glob.glob(join(iter_folder, 'r*'))

    # Loop over each main mc iteration.  All subfolders in iteration span the
    # parameter space
    for i, fdir in enumerate(iter_folder_list):

        print 'Compiling GMT: file {:02d}/{:02d}'.format(i+1,
                                                         len(iter_folder_list))
        times, gmts = compile_gmts(fdir, a_d_vals=a_d_vals)
        gmts = center_to_time_range(times, gmts, -1, trange=center_trange)

        if i == 0:
            gmt_out = np.zeros((len(iter_folder_list), gmts.shape[0],
                                gmts.shape[1]))

        gmt_out[i] = gmts

    ce_vals = np.zeros(gmt_out.shape[:2])
    r_vals = np.zeros_like(ce_vals)

    for gmt_iter, ce_val, r_val in izip(gmt_out, ce_vals, r_vals):
        tmp_ce, tmp_r = ce_r_ens_avg(gmt_iter, times, analysis_gmt,
                                     analysis_times, trange=trange,
                                     center_trange=center_trange)
        ce_val[:] = tmp_ce
        r_val[:] = tmp_r

    return times, gmt_out, ce_vals, r_vals


def ce_r_ens_avg(gmt_ens, times, analysis_gmt, analysis_times,
                 trange=(1880, 2000), center_trange=(1900, 2000)):

    start, end = trange
    # Get the global mean for the analysis dataset

    gmt_ens = center_to_time_range(times, gmt_ens, time_axis=-1,
                                   trange=center_trange)

    # Get a mask for overlapping times
    analysis_tidx = (analysis_times >= start) & (analysis_times <= end)
    lmr_tidx = (times >= start) & (times <= end)

    ens_ce = \
        np.array([utils2.coefficient_efficiency(analysis_gmt[analysis_tidx],
                                                a_ens_gmt[lmr_tidx])
                  for a_ens_gmt in gmt_ens])
    ens_ce[ens_ce == 1] = np.nan

    ens_r = np.array([np.corrcoef(analysis_gmt[analysis_tidx],
                                  a_ens_gmt[lmr_tidx])[0, 1]
                      for a_ens_gmt in gmt_ens])

    return ens_ce, ens_r


def detrend_data(X, y, ret_coef=True):

    model = linear_model.LinearRegression()
    model.fit(X, y)
    linfit_line = model.predict(X)

    if ret_coef:
        return linfit_line, {'slope': model.coef_,
                             'intercept': model.intercept_}

    else:
        return linfit_line


def calc_analysis_gmt(analysis_var_obj, trange=(1880,2000),
                      center_trange=(1900,2000),
                      detrend=False):
    gm_analysis = utils2.global_mean2(analysis_var_obj.data,
                                      analysis_var_obj.lat)
    gm_analysis = center_to_time_range(analysis_var_obj.time, gm_analysis, 0,
                                       trange=center_trange)
    analysis_tidx = ((analysis_var_obj.time >= trange[0]) &
                     (analysis_var_obj.time <= trange[1]))
    gm_analysis = gm_analysis[analysis_tidx]

    if detrend:
        linfit_line, coef = detrend_data(
                analysis_var_obj.time[analysis_tidx][:, None],
                gm_analysis[:, None],
                ret_coef=True)
        gm_analysis -= linfit_line.squeeze()

    return gm_analysis


def trends_ens_avg(gmt_ens, times, trange=(1880, 2000), ret_detrend_ens=False):

    start, end = trange
    time_idx = (times >= start) & (times <= end)
    gmt_ens = gmt_ens[:, time_idx]
    trends = np.empty(len(gmt_ens)) * np.nan
    times = times[time_idx]

    # Get non-NaNs
    mask = (~np.isfinite(gmt_ens)).sum(axis=1) == 0
    mask_gmt_ens = gmt_ens[mask]

    linfit_line, coefs = detrend_data(times[:, None], mask_gmt_ens.T)

    trends[mask] = coefs['slope'].squeeze()

    if ret_detrend_ens:
        detr_ens = np.empty_like(gmt_ens) * np.nan
        detr_ens[mask] = mask_gmt_ens - linfit_line.T

        return trends, times, detr_ens

    return trends,


def load_from_paramsearch_exp(exp_dir, out_dir, out_fname, ignore_npz,
                              analysis_gmt, analysis_times, a_vals, d_vals,
                              trange=(1880, 2000), center_trange=(1900, 2000)):
    if not exists(join(out_dir, out_fname)) or ignore_npz:

        a_d_vals = [result for result in product(a_vals, d_vals)]

        [times, gmt_vals,
         ce_vals, r_vals] = compile_iters(exp_dir,
                                          analysis_gmt,
                                          analysis_times,
                                          a_d_vals,
                                          trange=trange,
                                          center_trange=center_trange)

        try:
            os.makedirs(out_dir)
        except OSError:
            pass

        result = {'gmt_vals': gmt_vals,
                  'ce_vals': ce_vals,
                  'r_vals': r_vals,
                  'times': times,
                  'a_vals': a_vals,
                  'd_vals': d_vals}

        np.savez(join(out_dir, out_fname), **result)

    else:
        npfile = np.load(join(out_dir, out_fname))
        result = dict(npfile)

    return result


def plot_heatmap(data, a_vals, d_vals, dat_type='CE', title='Title',
                 xlabel='$d$ values', ylabel='$a$ values', savefig=False,
                 savefile=None):

    if dat_type == 'CE':
        vmin, vmax = [-1, 1]
    elif dat_type == 'corr':
        vmin, vmax = [0, 1]

    df = pd.DataFrame(data=data, columns=d_vals, index=a_vals)

    sns.heatmap(df.iloc[::-1], vmin=vmin, vmax=vmax, annot=True, fmt='1.2f')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if savefig:
        plt.savefig(savefile)

    plt.show()


def plot_baseline_compare(x_vals, data, baseline, title='Title',
                          xlabel='$a$ values', ylabel='CE', savefig=False,
                          savefile=None, show=True, legend_labels=None):

    plt.hlines(baseline, xmin=x_vals[0], xmax=x_vals[-1], linewidth=2,
               color='k', linestyles='dashed', label='production baseline')

    for line in data:
        plt.plot(x_vals, line, linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if legend_labels is not None:
        plt.legend(legend_labels, loc='lower left')

    plt.tight_layout()

    if savefig:
        plt.savefig(savefile)

    if show:
        plt.show()


def main():

    # general params
    trange = (1880, 2000)
    center_trange = (1900, 2000)
    ignore_npz = False
    do_detrended = True
    sns.set_style(style='whitegrid')
    a_vals = np.arange(0.5, 1, 0.1)
    d_vals = np.array([0])
    sns.set_context('talk', font_scale=1.5)

    # directories
    data_dir = '/home/disk/kalman2/wperkins/LMR_output/testing'
    exp_name = ['testdev_paramsearch_xbblend_a0a9pt2_d0_51itr',
                'testdev_paramsearch_noxbblend_a0a9pt2_d0_25itr']
    exp_name = ['testdev_paramsearch_infl_1pt1',
                'testdev_paramsearch_infl_1pt4']
    exp_dirs = [join(data_dir, exp) for exp in exp_name]

    production_dir = '/home/disk/kalman2/wperkins/LMR_output/archive'
    prod_exp_name = 'production_gis_ccsm4_pagesall_0.75'

    save_fig = True
    out_dirs = [join(exp_dir, 'paramsearch_out') for exp_dir in exp_dirs]

    center_tag = 'c{:4d}_{:4d}'.format(*center_trange)
    trange_tag = 't{:4d}_{:4d}'.format(*trange)
    out_fname = 'ce_r_gmt_' + trange_tag + '_' + center_tag + '.npz'

    # load analysis (TODO: needs to be GISTEMP in config for now)
    LMR_config.psm.linear.sub_base_res = 1.0
    LMR_config.core.assimilation_time_res = [1.0]
    grid_class = lmrgrid.get_analysis_var_class('GISTEMP')
    gistemp = grid_class.load(LMR_config.psm.linear)[0]
    gm_analysis = calc_analysis_gmt(gistemp, center_trange=center_trange)
    detrended_gm_analysis = calc_analysis_gmt(gistemp,
                                              center_trange=center_trange,
                                              detrend=True)
    analysis_tidx = (gistemp.time >= trange[0]) & (gistemp.time <= trange[1])
    gm_times = gistemp.time[analysis_tidx]
    gm_analysis = gm_analysis[analysis_tidx]
    detrended_gm_analysis = detrended_gm_analysis[analysis_tidx]

    # result_dicts = []
    # for exp_dir, out_dir in zip(exp_dirs, out_dirs):
    #     result_dicts.append(load_from_paramsearch_exp(exp_dir, out_dir,
    #                                                   out_fname, ignore_npz,
    #                                                   gm_analysis, gm_times,
    #                                                   a_vals, d_vals,
    #                                                   trange=trange,
    #                                                   center_trange=center_trange))
    #
    # # A-values ensemble mean GMT
    # psearch_avggmt_times = []
    # for result in result_dicts:
    #     gmts = result['gmt_vals']
    #     gmt_avg = np.nanmean(gmts, axis=0)
    #     num_nans = (~np.isfinite(result['ce_vals'])).sum(axis=0)
    #     nan_pct = num_nans/float(len(gmts))
    #     gmt_avg[nan_pct > 0.25] = np.nan
    #     psearch_avggmt_times.append((gmt_avg, result['times']))
    #
    # ce_r_values = [ce_r_ens_avg(tup[0], tup[1], gm_analysis, gm_times,
    #                             trange=trange,
    #                             center_trange=center_trange)
    #                for tup in psearch_avggmt_times]
    #
    # trend_res = [trends_ens_avg(tup[0], tup[1], trange=trange,
    #                             ret_detrend_ens=do_detrended)
    #              for tup in psearch_avggmt_times]
    #
    # if do_detrended:
    #     trends, detr_times, detr_ens = zip(*trend_res)
    #     dtr_ce_r_values = [ce_r_ens_avg(tmp_ens, tmp_time,
    #                                     detrended_gm_analysis, gm_times,
    #                                     trange=trange,
    #                                     center_trange=center_trange)
    #                        for tmp_ens, tmp_time in izip(detr_ens,
    #                                                      detr_times)]
    #
    #     detr_ens_ce, detr_ens_r = zip(*dtr_ce_r_values)
    # else:
    #     trends, = zip(*trend_res)
    #
    # ens_ce, ens_r = zip(*ce_r_values)
    # # plot_heatmap(ens_ce, a_vals, d_vals)
    #
    # title = ('Hybrid DA {} Comparison (' + trange_tag + '_' + center_tag +
    #          ')')
    # leg = ['Blended Cov. & Prior', 'Blended Cov. Only', 'Ref. Gistemp']
    #
    # # Production baseline
    # prod_times, prod_gmt = compile_gmts(join(production_dir, prod_exp_name))
    # prod_ens_gmt = prod_gmt.mean(axis=0, keepdims=True)
    # prod_ce, prod_r = ce_r_ens_avg(prod_ens_gmt, prod_times, gm_analysis,
    #                                gm_times, trange=trange,
    #                                center_trange=center_trange)
    #
    # prod_trend, detr_prod_times, detr_prod_ens_gmt = \
    #     trends_ens_avg(prod_ens_gmt, prod_times, trange=trange,
    #                    ret_detrend_ens=True)
    #
    # detr_prod_ce, detr_prod_r = ce_r_ens_avg(detr_prod_ens_gmt,
    #                                          detr_prod_times,
    #                                          detrended_gm_analysis,
    #                                          gm_times,
    #                                          trange=trange,
    #                                          center_trange=center_trange)
    #
    # plot_baseline_compare(a_vals, list(ens_ce), prod_ce,
    #                       xlabel='blending coefficient',
    #                       legend_labels=leg,
    #                       title=title.format('CE'))
    # rstr = 'correlation'
    # plot_baseline_compare(a_vals, list(ens_r), prod_r, ylabel=rstr,
    #                       xlabel='Blending Coefficient',
    #                       legend_labels=leg,
    #                       title=title.format(rstr))
    # tstr = 'Trend'
    # plot_baseline_compare(a_vals, trends, prod_trend,
    #                       xlabel='Blending Coefficcient',
    #                       legend_labels=leg,
    #                       title=title.format(tstr),
    #                       ylabel='Trend (K/yr)')
    #
    # if do_detrended:
    #     plot_baseline_compare(a_vals, list(detr_ens_ce), detr_prod_ce,
    #                           xlabel='blending coefficient',
    #                           legend_labels=leg,
    #                           title=(title.format('CE') + ' (detrended)'))
    #     rstr = 'correlation'
    #     plot_baseline_compare(a_vals, list(detr_ens_r), detr_prod_r,
    #                           ylabel=rstr,
    #                           xlabel='Blending Coefficient',
    #                           legend_labels=leg,
    #                           title=(title.format(rstr) + ' (detrended)'))


if __name__ == '__main__':
    main()
