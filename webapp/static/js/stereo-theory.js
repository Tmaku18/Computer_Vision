const $ = (id) => document.getElementById(id);

$("btn-numbers").addEventListener("click", async () => {
  $("numbers").textContent = "Computing…";
  const form = new FormData();
  form.set("sample", "1");
  form.set("baseline", "0.2");
  form.set("unit", "m");
  const res = await fetch("/api/m7/uncalibrated", { method: "POST", body: form });
  const body = await res.json();
  $("numbers").textContent = res.ok ? body.text : (body.detail || "Failed.");
});
