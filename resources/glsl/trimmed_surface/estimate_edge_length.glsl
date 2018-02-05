
///////////////////////////////////////////////////////////////////////////////
float estimate_control_edge(in samplerBuffer points,
  in int first_point_index,
  in int second_point_index,
  in int stride,
  in int limit)
{
  int i = first_point_index;
  int j = second_point_index;

  float edge_length = 0;

  for (; j < limit; i += stride, j += stride) {
    // get point in hyperspace coordinates (wx, wy, wz, w)
    vec4 p0 = texelFetchBuffer(points, i);
    vec4 p1 = texelFetchBuffer(points, j);

    // normalize to euclidian
    p0 /= p0.w;
    p1 /= p1.w;

    vec3 edge = p0.xyz - p1.xyz;
    edge_length += length(edge);
  }
  return edge_length;
}


///////////////////////////////////////////////////////////////////////////////
float estimate_control_edge_in_pixel(in samplerBuffer points,
  in int first_point_index,
  in int second_point_index,
  in int stride,
  in int limit,
  in mat4 mvp,
  in vec2 resolution)
{
  int i = first_point_index;
  int j = second_point_index;

  float edge_length = 0;

  for (; j < limit; i += stride, j += stride) {
    // get point in hyperspace coordinates (wx, wy, wz, w)
    vec4 p0 = texelFetchBuffer(points, i);
    vec4 p1 = texelFetchBuffer(points, j);

    // normalize to euclidian
    p0 /= p0.w;
    p1 /= p1.w;

    // transform to screen coordinates
    p0 = mvp * vec4(p0.xyz, 1.0);
    p1 = mvp * vec4(p1.xyz, 1.0);

    p0 /= p0.w;
    p1 /= p1.w;

    vec2 p0_device = (p0.xy * 0.5 + 0.5) * resolution;
    vec2 p1_device = (p1.xy * 0.5 + 0.5) * resolution;

    vec2 edge = p0_device.xy - p1_device.xy;
    edge = clamp(edge, vec2(0), resolution);
    edge_length += length(edge);
  }
  return edge_length;
}

///////////////////////////////////////////////////////////////////////////////
vec4 estimate_edge_lengths_in_pixel(in int base_id,
  in samplerBuffer points,
  in int order_u,
  in int order_v,
  in mat4 mvp,
  in vec2 resolution)
{
  float edge_length_umin = estimate_control_edge_in_pixel(points,
    base_id,
    base_id + order_u,
    order_u,
    base_id + order_u * order_v,
    mvp,
    resolution);
  float edge_length_umax = estimate_control_edge_in_pixel(points,
    base_id + (order_u - 1),
    base_id + (2 * order_u - 1),
    order_u,
    base_id + order_u * order_v,
    mvp,
    resolution);

  float edge_length_vmin = estimate_control_edge_in_pixel(points,
    base_id + order_u * (order_v - 1),
    base_id + order_u * (order_v - 1) + 1,
    1,
    base_id + order_u * order_v,
    mvp,
    resolution);

  float edge_length_vmax = estimate_control_edge_in_pixel(points,
    base_id,
    base_id + 1,
    1,
    base_id + order_u,
    mvp,
    resolution);

  return vec4(edge_length_umin, edge_length_umax, edge_length_vmin, edge_length_vmax);

}
