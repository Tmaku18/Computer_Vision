Put the two phone clips here. They are gitignored.

Video A: phone propped still, people or cars moving, 1080p, 30 fps, trimmed to 30–40 s.
Video B: walk forward or pan, same length.

Then, from the repository root:

  .venv/bin/python HW4/optical_flow_video.py --video HW4/videos/a.mov --duration 30 --width 720
  .venv/bin/python HW4/validate_tracking.py --video HW4/videos/a.mov --time 3 --clicks clicks.json --save --width 720

The hosted pages play whatever is committed under HW4/results/. Until those phone clips exist, HW4/make_demo_results.py writes a short synthetic stand-in and labels it as such.
