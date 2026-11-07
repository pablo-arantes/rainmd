/**
 * LogMD's upload format, and a share that can fail honestly.
 *
 * 🔴 THIS SENDS A STRUCTURE TO A THIRD PARTY, which is the one thing the rest of
 * this program is built not to do. It is opt-in per run, it is never called by
 * anything automatic, and the button that calls it says so. Nothing else on the
 * page makes a network request at all.
 *
 * 🔴 AND THE ENDPOINT IS NOT RESPONDING AT THE TIME OF WRITING. Measured on
 * 2026-11-07, from a server where CORS does not apply, every published LogMD
 * backend URL returns `404 modal-http: invalid function call`:
 *
 *   alexander-mathiasen--logmd-upload-frame.modal.run       404
 *   alexander-mathiasen--logmd-upload-frame-bcif.modal.run  404
 *   alexander-mathiasen--logmd-list-project.modal.run       404
 *
 * Those are the URLs in logmd 0.1.46 on PyPI, not stale ones from a clone - the
 * published client points at the same places. So the upload path here is
 * written to their documented payload and cannot currently be verified against
 * a live service. That is why `shareRun` reports the exact status it got rather
 * than a generic failure, and why `logmdBundle` exists: the bundle is the same
 * frames in the same format, downloadable, so the official `logmd` client can
 * upload them whenever the service answers again. The half that works today is
 * the half that does not depend on somebody else's infrastructure.
 *
 * 🔴 NO TOKEN, NO EMAIL, NO ACCOUNT. LogMD's client sends `user_id: token.email`
 * when logged in and `"public"` when not. This only ever sends `"public"` with a
 * null token - a page has no business asking for someone's LogMD credentials,
 * and an anonymous public run is what the share is for.
 */

/** LogMD's own id shape for an anonymous run: ten hex characters. */
export function logmdRunId(random = Math.random) {
  let id = "";
  while (id.length < 10) id += Math.floor(random() * 16).toString(16);
  return id.slice(0, 10);
}

/**
 * One frame, in the shape `upload_worker_process` posts.
 *
 * @param {object} options
 * @param {string} options.runId
 * @param {number} options.frame zero-based frame number
 * @param {string} options.pdb the frame as a PDB block
 * @param {object} [options.values] scalars to plot beside the structure; LogMD
 *   calls this `data_dict` and plots whatever it is given - it computes nothing
 *   itself, which is why `shared/analysis/metrics.js` still has to exist.
 */
export function logmdFrame(options) {
  return {
    user_id: "public",
    run_id: options.runId,
    frame_num: String(options.frame),
    file_contents: options.pdb,
    token: null,
    project: "",
    data_dict: options.values ?? {},
  };
}

/** Where an anonymous run would be viewable, if the upload succeeded. */
export const logmdViewUrl = (runId) => `https://rcsb.ai/${runId}`;

/** Their published upload endpoint. */
export const LOGMD_UPLOAD_URL =
  "https://alexander-mathiasen--logmd-upload-frame.modal.run";

/**
 * Post every frame, reporting precisely what happened.
 *
 * 🔴 SEQUENTIAL, NOT PARALLEL. Their own client runs three upload workers, but
 * a page firing a hundred concurrent cross-origin POSTs at a serverless
 * endpoint is indistinguishable from an attack and will be rate-limited. One at
 * a time is slower and is the only polite option from a browser.
 *
 * @returns {Promise<{ok: boolean, runId: string, url: string, sent: number,
 *   total: number, failure: string|null}>}
 */
export async function shareRun(frames, options = {}) {
  const runId = options.runId ?? logmdRunId();
  const fetchImpl = options.fetch ?? globalThis.fetch;
  const onProgress = options.onProgress ?? (() => {});
  let sent = 0;
  let failure = null;
  for (let index = 0; index < frames.length; index += 1) {
    const body = logmdFrame({ runId, frame: index, pdb: frames[index].pdb,
      values: frames[index].values });
    let response;
    try {
      response = await fetchImpl(options.url ?? LOGMD_UPLOAD_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
    } catch (error) {
      // 🔴 A CROSS-ORIGIN FAILURE IS INDISTINGUISHABLE FROM A DEAD SERVER HERE.
      // `fetch` rejects with the same opaque "Failed to fetch" whether the
      // endpoint refused the preflight or does not exist, because the browser
      // will not say which - so the message names both possibilities rather
      // than guessing one.
      failure = `the browser could not reach LogMD (${error.message}). That is `
        + "either CORS - their endpoint not permitting a request from this "
        + "page - or the endpoint being down. A browser cannot tell the two "
        + "apart. Measured 2026-11-07, it returns 404 from a server too.";
      break;
    }
    if (!response.ok) {
      failure = `LogMD answered HTTP ${response.status} on frame ${index}`;
      break;
    }
    sent += 1;
    onProgress(sent, frames.length);
  }
  return {
    ok: failure === null && sent === frames.length,
    runId,
    url: logmdViewUrl(runId),
    sent,
    total: frames.length,
    failure,
  };
}

/**
 * The same frames as a downloadable bundle, for the official client to upload.
 *
 * 🔴 THE HALF THAT DOES NOT DEPEND ON ANYONE ELSE. One JSON file holding the
 * exact payloads, plus the three lines of Python that post them - so a reader
 * whose share failed still has their run in LogMD's format and a documented way
 * in, rather than a dead button.
 */
export function logmdBundle(frames, runId) {
  const payloads = frames.map((frame, index) => logmdFrame({
    runId, frame: index, pdb: frame.pdb, values: frame.values,
  }));
  return JSON.stringify({
    note: "Frames in LogMD's upload format. RainMD could not post them itself; "
      + "see `how` below. Produced by RainMD, which runs entirely in the browser.",
    run_id: runId,
    view_url: logmdViewUrl(runId),
    upload_url: LOGMD_UPLOAD_URL,
    how: [
      "pip install httpx",
      "python - <<'EOF'",
      "import json, httpx",
      "bundle = json.load(open('logmd-frames.json'))",
      "for frame in bundle['frames']:",
      "    r = httpx.post(bundle['upload_url'], json=frame, timeout=180)",
      "    print(frame['frame_num'], r.status_code)",
      "print(bundle['view_url'])",
      "EOF",
    ],
    frames: payloads,
  }, null, 1);
}
