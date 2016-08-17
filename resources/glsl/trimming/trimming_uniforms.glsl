#ifndef GPUCAST_GLSL_TRIMMING_UNIFORMS
#define GPUCAST_GLSL_TRIMMING_UNIFORMS

// fast pre-classification texture
uniform usamplerBuffer gpucast_preclassification;

/*******************************************************************************
 *  Trimming for parameter pair [u,v] : DATA STRUCTURE :
 *
 *  - gpucast_bp_trimdata :
 *      [index]
 *      trim_index          : [vmin_all, vmax_all, #tdata, 0.0]
 *      trim_index + 1      : [vmin_clist, vmax_clist, urangeslist_id, 0.0]
 *      ...                    ...
 *      trim_index + #tdata : [vmin_clist, vmax_clist, urangeslist_id, 0.0]
 *
 *  - gpucast_bp_celldata :
 *      [index]
 *      urangeslist_id                  : [umin_all, umax_all, nr_of_uranges, 0.0]
 *      urangeslist_id + 1              : [umin,     umax, intersect_on_right, curvelist_id]
 *             ...
 *      urangeslist_id + nr_of_uranges  : [umin,     umax, intersect_on_right, curvelist_id]
 *
 *  -gpucast_bp_curvelist :
 *      [index]
 *      curvelist_id                    : [nr_of_curves]
 *      curvelist_id + 1                : [curve_id, curveorder, increasing, 0.0]
 *      ...
 *      curvelist_id + nr_of_curves     : [curve_id, curveorder, increasing, 0.0]
 *
 *  - gpucast_bp_curvedata :
 *      [index]
 *      curve_id            : [x0, y0, w0, 0.0]
 *             ...
 *      curve_id + order    : [xn, yn, wn, 0.0]
 *******************************************************************************/

// classic double-binary monotonic curve partition
uniform samplerBuffer gpucast_bp_trimdata;
uniform samplerBuffer gpucast_bp_celldata;
uniform samplerBuffer gpucast_bp_curvelist;
uniform samplerBuffer gpucast_bp_curvedata;

// binary partitioned contours 
uniform samplerBuffer gpucast_cmb_partition;
uniform samplerBuffer gpucast_cmb_contourlist;
uniform samplerBuffer gpucast_cmb_curvelist;
uniform samplerBuffer gpucast_cmb_curvedata;
uniform samplerBuffer gpucast_cmb_pointdata;

// kd-contour-based trimming data structures
uniform samplerBuffer gpucast_kd_partition;
uniform samplerBuffer gpucast_kd_contourlist;
uniform samplerBuffer gpucast_kd_curvelist;
uniform samplerBuffer gpucast_kd_curvedata;
uniform samplerBuffer gpucast_kd_pointdata;

#endif