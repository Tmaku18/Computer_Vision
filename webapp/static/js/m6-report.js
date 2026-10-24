import { $, api } from "/static/js/api.js";

const KEYS = { links: "m6.links", flow: "m6.flow", tracking: "m6.tracking", sfm: "m6.sfm" };

function read(key) {
  try {
    return JSON.parse(localStorage.getItem(key) || "null");
  } catch {
    return null;
  }
}

function saveLinks() {
  const meta = { gh: $("#gh").value, app: $("#app").value, video: $("#video").value };
  localStorage.setItem(KEYS.links, JSON.stringify(meta));
  const bits = [
    meta.gh && `<div>GitHub: <a href="${meta.gh}">${meta.gh}</a></div>`,
    meta.app && `<div>App: <a href="${meta.app}">${meta.app}</a></div>`,
    meta.video && `<div>Video: <a href="${meta.video}">${meta.video}</a></div>`,
  ].filter(Boolean);
  $("#links").innerHTML = bits.join("") || "Add the GitHub, app, and video links above. They are saved in this browser.";
}

["gh", "app", "video"].forEach((id) => $("#" + id).addEventListener("input", saveLinks));
const saved = read(KEYS.links) || {};
$("#gh").value = saved.gh || "";
$("#app").value = saved.app || "";
$("#video").value = saved.video || "";
saveLinks();

function mean(values) {
  const ok = (values || []).filter((v) => v !== null && Number.isFinite(Number(v)));
  if (!ok.length) return null;
  return ok.reduce((sum, v) => sum + Number(v), 0) / ok.length;
}

function flowBlock(flow) {
  const foe = flow.focus_of_expansion || {};
  const title = flow.label ? `<h3>${flow.label}</h3>` : "";
  return `${title}
    <p>${flow.interpretation || ""}</p>
    <p class="small">${flow.note || ""}</p>
    <p class="small">${flow.frames || "?"} frames, mean speed ${(flow.mean_speed_average || 0).toFixed(2)} px/frame,
    moving fraction ${((flow.moving_fraction_mean || 0) * 100).toFixed(1)}%.
    Focus of expansion: ${foe.x == null ? "not estimated" : foe.x.toFixed(0) + ", " + foe.y.toFixed(0)}.</p>`;
}

function trackingBlock(item) {
  const parts = [
    ["from-scratch", mean(item.scratch_error)],
    ["OpenCV", mean(item.opencv_error)],
    ["template", mean(item.template_error)],
  ].filter(([, v]) => v !== null).map(([name, v]) => `${name} mean error ${v.toFixed(3)} px`);
  const summary = item.summary || parts.join(" · ");
  return `<h3>${item.label || "Tracking"}</h3>
    <p>${summary || ""}</p>
    <p class="small">${item.note || ""}</p>
    <pre class="matrix">${item.example_text || ""}</pre>`;
}

function sfmBlock(sfm) {
  if (!sfm || !sfm.K) return "";
  const measured = sfm.camera_positions || [];
  const rows = (sfm.cameras || []).map((cam, i) => {
    const tape = measured[i];
    const tapeCell = tape
      ? [tape.x_m, tape.y_m, tape.z_m].map((v) => Number(v).toFixed(3)).join(", ")
      : "—";
    return `<tr>
      <td>${i + 1}</td>
      <td class="num">${cam.center.map((v) => Number(v).toFixed(3)).join("</td><td class=\"num\">")}</td>
      <td class="num">${tapeCell}</td>
      <td class="num">${Number(sfm.reproj_rms_after[i]).toFixed(3)}</td>
    </tr>`;
  }).join("");
  const edges = (sfm.polygon_unit.edge_lengths || []).map((v) => Number(v).toFixed(3)).join(", ");
  const check = sfm.check
    ? `<p>Second tape ${sfm.check.tape} ${sfm.check.unit}, reconstruction ${Number(sfm.check.reconstructed).toFixed(3)}, difference ${Number(sfm.check.error).toFixed(3)}.</p>`
    : "";
  return `
    <p class="small">${sfm.note || ""}</p>
    <p class="small">K (pixels), image ${sfm.image_size}, ${sfm.n_sift_tracks} SIFT tracks plus ${sfm.n_boundary} boundary points.</p>
    <pre class="matrix">${sfm.K.map((row) => row.map((v) => Number(v).toFixed(2)).join("  ")).join("\n")}</pre>
    <p>Distortion (k1, k2, p1, p2, k3): ${(sfm.dist_coeffs || []).map((v) => Number(v).toFixed(4)).join(", ")}</p>
    <p class="small">Recovered centers are in meters relative to view 1, with Y pointing down. Tape positions are in the object's frame (the poster at the origin, Y up), so the two columns are the same scene written in different coordinates.</p>
    <table>
      <thead><tr><th>View</th><th class="num">Recovered x</th><th class="num">y</th><th class="num">z</th><th class="num">Tape x, y, z (m)</th><th class="num">RMS px</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <p>Boundary edges (${sfm.polygon_unit.unit}): ${edges}. Area ${Number(sfm.polygon_unit.area).toFixed(3)}.</p>
    ${check}
    <pre class="matrix">${sfm.workout || ""}</pre>`;
}

function localSection() {
  const bits = [];
  const flow = read(KEYS.flow);
  const tracking = read(KEYS.tracking);
  const sfm = read(KEYS.sfm);
  if (flow) bits.push(flowBlock(flow));
  if (tracking) bits.push(trackingBlock(tracking));
  if (sfm) bits.push(sfmBlock(sfm));
  if (!bits.length) return;
  $("#local-card").hidden = false;
  $("#local-body").innerHTML = bits.join("");
}

api("/api/m6/results").then((catalog) => {
  const clips = catalog.clips || [];
  if (clips.length) $("#flow-body").innerHTML = clips.map(flowBlock).join("");
  const pairs = catalog.tracking || [];
  if (pairs.length) $("#track-body").innerHTML = pairs.map(trackingBlock).join("");
  const sfm = catalog.sfm;
  if (sfm && sfm.result) {
    const views = sfm.views || [];
    $("#sfm-views").innerHTML = views.map((src, i) =>
      `<figure><img src="${src}" alt="View ${i + 1}" style="width:100%;border-radius:8px"><figcaption class="small">View ${i + 1}</figcaption></figure>`
    ).join("");
    $("#sfm-body").innerHTML = sfmBlock(sfm.result);
  }
  localSection();
}).catch(() => localSection());
