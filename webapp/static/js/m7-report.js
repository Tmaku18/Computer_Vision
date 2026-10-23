const $ = (id) => document.getElementById(id);
const repo = $("repo-url");
const app = $("app-url");
const video = $("video-url");
app.value = location.origin + "/module7";

function sync() {
  const set = (id, input) => {
    const el = $(id);
    const value = input.value.trim();
    el.textContent = value || "—";
    el.href = value || "#";
  };
  set("r-repo", repo);
  set("r-app", app);
  set("r-video", video);
}
[repo, app, video].forEach((el) => el.addEventListener("input", sync));
sync();
$("r-date").textContent = new Date().toLocaleDateString(undefined, { year: "numeric", month: "long", day: "numeric" });
$("btn-print").addEventListener("click", () => window.print());

function cm(value) {
  return value == null ? "—" : `${(Number(value) * 100).toFixed(2)} cm`;
}

fetch("/api/m7/report").then((res) => res.json()).then((body) => {
  $("r-note").textContent = body.sample_note || "";
  $("r-rows").innerHTML = body.rows.map((row) => {
    const est = row.estimated;
    const truth = row.truth;
    let estimated = "";
    let known = "";
    if (row.shape === "rectangle") {
      estimated = `width ${cm(est.width_m)}, height ${cm(est.height_m)}`;
      known = `width ${cm(truth.width_m)}, height ${cm(truth.height_m)}`;
    } else if (row.shape === "circle") {
      estimated = `diameter ${cm(est.diameter_m)}`;
      known = `diameter ${cm(truth.diameter_m)}`;
    } else {
      estimated = est.edges_m.map((edge) => cm(edge)).join(", ");
      known = truth.edges_m.map((edge) => cm(edge)).join(", ");
    }
    return `<tr><td>${row.shape}</td><td>${estimated}</td><td>${known}</td></tr>`;
  }).join("");
  const room = body.room;
  $("r-room").textContent = `Synthetic box: length ${room.length_m.toFixed(3)} m, width ${room.width_m.toFixed(3)} m, height ${room.height_m.toFixed(3)} m `
    + `(truth ${body.room_truth.length_m} × ${body.room_truth.width_m} × ${body.room_truth.height_m} m). `
    + `Height from the long wall ${room.height_from_long_wall_m.toFixed(3)} m, from the short wall ${room.height_from_short_wall_m.toFixed(3)} m.`;
  $("r-refs").innerHTML = body.references.map((ref) => `<li>${ref}</li>`).join("");
});
