/**
 * Drive the page in a real browser over the Chrome DevTools Protocol.
 *
 * 🔴 THIS LANE IS THE SHIPPING TARGET AND THE NODE LANE IS NOT. Dawn's node
 * binding and Chrome's WebGPU are different implementations of one spec, and
 * they disagree about exactly the things that bite: whether an out-of-range
 * store is clamped or discarded, which optional features exist, how a lost
 * device is reported. A kernel validated only through `tools/gpu-node.mjs` has
 * been validated on an implementation no user runs.
 *
 *   node tools/gpu-page.mjs                               # smoke test the page
 *   node tools/gpu-page.mjs --check                       # drive the page's own GPU button
 *   node tools/gpu-page.mjs --module=tools/gpu/check-nonbonded.mjs
 *                                                         # run a check module IN CHROME
 *   node tools/gpu-page.mjs --simulate                    # drive a real GPU run through the UI
 */
import { spawn } from "node:child_process";
import { setTimeout as delay } from "node:timers/promises";
import { statSync, writeFileSync } from "node:fs";

const CHROME_CANDIDATES = [
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/Applications/Chromium.app/Contents/MacOS/Chromium",
  "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
];

/**
 * 🔴 PER-PROCESS PORTS, BECAUSE THESE RUN BACK TO BACK. Both were fixed - 4188
 * and 9333 - and `chrome.kill()` is asynchronous, so the dying browser from the
 * previous invocation still held the debugging port when the next one asked for
 * it. The symptom is the eleventh link of an `&&` chain failing with "the page
 * never assigned window.rainmd" while every one of the eleven passes when run
 * alone, which reads as a flaky page and is a flaky harness. `process.pid` is
 * unique among processes that exist at the same time, which is exactly the
 * guarantee a chain needs.
 */
const remoteArg = process.argv.find((argument) => argument.startsWith("--url="));
const PORT = 4100 + (process.pid % 500);
const DEBUG_PORT = 9100 + (process.pid % 500);
const PAGE_URL = remoteArg === undefined
  ? `http://127.0.0.1:${PORT}/` : remoteArg.slice("--url=".length);
const runCheck = process.argv.includes("--check");

/**
 * 🔴 A REJECTED PAGE PROMISE COMES BACK AS A RESULT, NOT AS AN ERROR. CDP puts
 * the thrown value in `result` and the stack in `exceptionDetails`, so with
 * `returnByValue` a rejection yields an OBJECT where a string was expected -
 * and `JSON.parse` on it reports `"[object Object]" is not valid JSON`, which
 * says nothing about what actually failed in the page. Checking
 * exceptionDetails first turns that into the real stack.
 */
function cdpValue(message, what) {
  const details = message.result?.exceptionDetails;
  if (details !== undefined) {
    const text = details.exception?.description ?? details.text ?? "unknown";
    throw new Error(`${what} threw in the page:\n${text}`);
  }
  const value = message.result?.result?.value;
  if (value === undefined) {
    throw new Error(`${what} returned nothing from the page`);
  }
  return value;
}

/**
 * Photograph an element, for any lane that wants one.
 *
 * 🔴 SHARED, BECAUSE IT WAS TRAPPED INSIDE `--simulate`. The `--view` lane
 * loads a structure fixture and never runs anything - the helix and strand are
 * vacuum peptides the GPU engine refuses - so the one lane that can show
 * secondary-structure colours could not take a picture of them.
 */
async function screenshot(socketUrl) {
  const shot = process.argv.find((argument) => argument.startsWith("--screenshot="));
  if (shot === undefined) return;
  const path = shot.slice("--screenshot=".length);
  const ofArg = process.argv.find((a) => a.startsWith("--shot-of="));
  const selector = ofArg === undefined ? "#viewer" : ofArg.slice("--shot-of=".length);
  const rect = await cdp(socketUrl, "Runtime.evaluate", {
    expression: `(() => {
      const node = document.querySelector(${JSON.stringify(selector)});
      if (node === null) return "null";
      node.scrollIntoView();
      const box = node.getBoundingClientRect();
      return JSON.stringify({ x: box.x + window.scrollX, y: box.y + window.scrollY,
        width: box.width, height: box.height });
    })()`,
    returnByValue: true,
  });
  const clip = JSON.parse(rect.result?.result?.value ?? "null");
  const capture = await cdp(socketUrl, "Page.captureScreenshot",
    clip === null ? { format: "png" }
      : { format: "png", captureBeyondViewport: true, clip: { ...clip, scale: 1 } });
  // 🔴 `result.data`, NOT `result.result.data`. Page.captureScreenshot returns
  // its payload directly where Runtime.evaluate wraps it in a RemoteObject;
  // reading the deeper path gave undefined and wrote no file, silently.
  const data = capture.result?.data;
  if (typeof data !== "string") return;
  const bytes = Buffer.from(data, "base64");
  writeFileSync(path, bytes);
  console.log(`screenshot:                  ${path} (${bytes.length} bytes)`);
}

async function cdp(url, method, params = {}) {
  const socket = new WebSocket(url);
  await new Promise((resolve, reject) => {
    socket.addEventListener("open", resolve, { once: true });
    socket.addEventListener("error", reject, { once: true });
  });
  const result = await new Promise((resolve, reject) => {
    socket.addEventListener("message", (event) => {
      const message = JSON.parse(event.data);
      if (message.id === 1) resolve(message);
    });
    socket.addEventListener("error", reject);
    socket.send(JSON.stringify({ id: 1, method, params }));
  });
  socket.close();
  return result;
}

const server = remoteArg === undefined
  ? spawn("python3", ["tools/serve.py", "--port", String(PORT)], { stdio: "ignore" })
  : { kill() {} };
// 🔴 `require` DOES NOT EXIST IN AN ES MODULE, and the catch swallowed the
// ReferenceError into "no Chrome found" - a probe reporting the absence of the
// thing it failed to look for. statSync is imported at the top instead.
const chromePath = CHROME_CANDIDATES.find((candidate) => {
  try { return statSync(candidate).isFile(); } catch { return false; }
});
if (chromePath === undefined) {
  server.kill();
  console.error("no Chrome/Chromium/Edge found; install one or run the page by hand");
  process.exit(2);
}

// 🔴 A SEPARATE PROFILE, ALWAYS. --enable-unsafe-webgpu turns off gates the
// browser set deliberately, and nobody's everyday browser should carry it.
/**
 * 🔴 WAIT FOR THE SERVER TO BIND BEFORE CHROME ASKS FOR THE PAGE. `serve.py`
 * was spawned and Chrome launched a few statements later, which is a race on
 * nothing more than how fast python starts: lose it and Chrome gets a
 * connection refused, shows its own error page, and never retries - so the
 * harness then waited the full thirty seconds for a `window.rainmd` that could
 * not arrive and reported "app.js threw during module evaluation", which is the
 * one message guaranteed to send the reader to the wrong file. Measured: one
 * invocation in three of a three-link chain.
 */
if (remoteArg === undefined) {
  let listening = false;
  for (let attempt = 0; attempt < 100; attempt += 1) {
    try {
      const response = await fetch(`http://127.0.0.1:${PORT}/index.html`,
        { method: "HEAD" });
      if (response.ok) { listening = true; break; }
    } catch { /* not up yet */ }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  if (!listening) {
    server.kill();
    console.error(`tools/serve.py never answered on port ${PORT}`);
    process.exit(2);
  }
}

const chrome = spawn(chromePath, [
  "--headless=new",
  `--remote-debugging-port=${DEBUG_PORT}`,
  // 🔴 THE WINDOW SIZE IS A LAUNCH FLAG, NOT A CDP OVERRIDE. `cdp()` opens a
  // socket per call and closes it, and an Emulation override lives for the
  // SESSION - so setting device metrics in one call and evaluating in the next
  // silently measured the default width. The layout lane reported 1200 px
  // whether it was asked for 1440 or 430, which reads as a page that ignores
  // its media queries and was the harness throwing the override away.
  // 2400 tall by default so a whole-page screenshot fits without scrolling.
  `--window-size=${(() => {
    const argument = process.argv.find((value) => value.startsWith("--width="));
    return argument === undefined ? 1200 : Number(argument.slice("--width=".length));
  })()},2400`,
  // 🔴 A FRESH PROFILE PER RUN. Chrome writes extension storage into the
  // profile with permissions the launching user cannot always remove, so a
  // fixed path accumulates undeletable directories and the next run inherits
  // whatever state the last one left.
  `--user-data-dir=/tmp/rainmd-chrome-${process.pid}`,
  "--enable-unsafe-webgpu",
  "--no-first-run",
  "--disable-gpu-sandbox",
  // 🔴 `--url=` POINTS THIS AT A DEPLOYED PAGE, which is the only way to check
  // the thing a visitor actually gets. Everything else here is served by
  // tools/serve.py from the working tree, so it validates the code on disk and
  // says nothing about what was published - a missing file, a wrong MIME type
  // or a path that only resolves at the server root are all invisible locally.
  PAGE_URL,
], { stdio: "ignore" });

try {
  // Wait for the debugging endpoint.
  let targets = null;
  for (let attempt = 0; attempt < 60; attempt += 1) {
    await delay(250);
    try {
      const response = await fetch(`http://127.0.0.1:${DEBUG_PORT}/json`);
      const list = await response.json();
      // Match the page we asked for, not the port - a remote URL has no port.
      const wanted = remoteArg === undefined ? String(PORT) : new URL(PAGE_URL).host;
      targets = list.filter((target) => target.type === "page"
        && target.url.includes(wanted));
      if (targets.length > 0) break;
    } catch { /* not up yet */ }
  }
  if (targets === null || targets.length === 0) throw new Error("Chrome never exposed the page target");
  const socketUrl = targets[0].webSocketDebuggerUrl;

  // 🔴 WAIT FOR THE PAGE'S OWN HANDLE, DO NOT SLEEP AND HOPE. This was
  // `await delay(2500)`, which is a guess about how long a module graph takes
  // to load - and the guess started failing the moment the graph grew, giving
  // `Cannot read properties of undefined (reading 'device')` on roughly one run
  // in three. That reads exactly like the bug where `web/app.js` threw during
  // module evaluation, which cost a whole session to diagnose, so it is worth
  // being unable to happen: `window.rainmd` is the last statement app.js runs,
  // so polling for it waits for exactly the right event.
  let ready = false;
  for (let attempt = 0; attempt < 120; attempt += 1) {
    const probe = await cdp(socketUrl, "Runtime.evaluate", {
      expression: "typeof window.rainmd", returnByValue: true,
    });
    if (probe.result?.result?.value === "object") { ready = true; break; }
    await delay(250);
  }
  if (!ready) {
    // 🔴 ONE RELOAD BEFORE GIVING UP. A page that lost any race during load
    // stays broken for ever - there is nothing in it to retry - so the only
    // recovery is to ask for it again. If the second attempt also fails, the
    // cause really is in the page.
    await cdp(socketUrl, "Page.navigate", { url: PAGE_URL });
    for (let attempt = 0; attempt < 80; attempt += 1) {
      await delay(250);
      const probe = await cdp(socketUrl, "Runtime.evaluate", {
        expression: "typeof window.rainmd", returnByValue: true,
      });
      if (probe.result?.result?.value === "object") { ready = true; break; }
    }
  }
  if (!ready) {
    throw new Error(
      "the page never assigned window.rainmd - web/app.js threw during module "
      + "evaluation, or a module failed to load. Check the browser console; "
      + "`npm test` has a static gate for the usual cause, a control named in "
      + "app.js that index.html does not have.",
    );
  }
  // The device probe is async even once the module is up.
  await delay(500);

  const probe = `(async () => {
    const report = document.getElementById("device-report").textContent;
    const errors = window.__rainmdErrors ?? [];
    return JSON.stringify({ title: document.title, deviceReport: report, errors });
  })()`;
  const result = await cdp(socketUrl, "Runtime.evaluate", {
    expression: probe, awaitPromise: true, returnByValue: true,
  });
  const payload = JSON.parse(result.result?.result?.value ?? "{}");
  console.log(`page title: ${payload.title}`);
  console.log(`device panel: ${payload.deviceReport?.replace(/\s+/g, " ").trim()}`);

  if (process.argv.includes("--simulate")) {
    const fixtureArg = process.argv.find((a) => a.startsWith("--fixture="));
    const fixtureName = fixtureArg === undefined
      ? "ala-wat" : fixtureArg.slice("--fixture=".length);
    // 🔴 THIS DRIVES THE CONTROLS, NOT THE API. The page's own handlers are
    // where the wiring lives - which engine is selected, whether the
    // trajectory writer is constructed, whether the download buttons become
    // enabled - and none of that is reachable by calling GpuSimulation
    // directly. Every bug this check has ever found was in a handler.
    const simulateExpression = `(async () => {
        const wait = (ms) => new Promise((r) => setTimeout(r, ms));
        // 🔴 RELATIVE, SO THIS LANE WORKS AGAINST A DEPLOYED PAGE TOO. A
        // leading slash resolves to the server root, which is the repository on
        // the dev server and the ACCOUNT root on a GitHub Pages project site -
        // so an absolute path passes locally and 404s on the only deployment
        // anyone will visit.
        const fixture = "FIXTURE_NAME";
        const [top, crd] = await Promise.all([
          fetch("test/fixtures/" + fixture + ".prmtop").then((r) => r.text()),
          fetch("test/fixtures/" + fixture + ".inpcrd").then((r) => r.text()),
        ]);
        const files = new DataTransfer();
        files.items.add(new File([top], fixture + ".prmtop"));
        files.items.add(new File([crd], fixture + ".inpcrd"));
        const input = document.getElementById("file-input");
        input.files = files.files;
        input.dispatchEvent(new Event("change"));
        await wait(2000);
        document.getElementById("engine").value = "gpu";
        document.getElementById("friction").value = "10";
        document.getElementById("pressure").value = "1";
        // 🔴 THE THREE STAGES, IN THE NOTEBOOKS' ORDER, and the durations are
        // times now rather than step counts. 0.008 ns at 2 fs is 4000 steps -
        // long enough that the pause window below falls INSIDE the run, which
        // is the condition that arm needs to mean anything.
        document.getElementById("min-steps").value = "1000";
        document.getElementById("eq-time").value = "0.008";
        // 1 ps = 500 steps at 2 fs, so an 8 ps run reports eight times - the
        // pause arm needs a step counter that has actually moved, and the
        // analysis needs at least two frames before RMSF means anything.
        document.getElementById("eq-write").value = "1";
        document.getElementById("eq-restraint").value = "500";

        document.getElementById("minimise").click();
        for (let i = 0; i < 120; i += 1) {
          await wait(250);
          if (!document.getElementById("minimise").disabled) break;
        }
        const equilibrateDisabled = document.getElementById("equilibrate").disabled;
        document.getElementById("equilibrate").click();

        // 🔴 PAUSE IS A GATE THE RUN LOOP AWAITS, AND ONLY THE BROWSER CAN SEE
        // IT HOLD. The handler sets a flag and the loop awaits a promise
        // BETWEEN blocks; nothing in the API can tell whether the integrator
        // actually stopped, or whether resuming continues from the same state
        // rather than restarting. So: wait for the first frames, pause, let
        // real time pass, and check the step counter has not moved.
        const stepText = () => {
          const cells = [...document.querySelectorAll("#live > div")];
          const cell = cells.find((c) => c.textContent.startsWith("step"));
          return cell === undefined ? "" : cell.textContent;
        };
        const running = () => !document.getElementById("stop").disabled;
        let binderFrozenDuringRun = null;
        let pauseHeld = null;
        for (let i = 0; i < 40; i += 1) {
          await wait(100);
          if (stepText() !== "") break;
        }
        if (stepText() !== "" && !document.getElementById("pause").disabled) {
          document.getElementById("pause").click();
          // Long enough that an unpaused run would report several more times.
          await wait(1200);
          const held = stepText();
          const activeAt = running();
          await wait(1200);
          binderFrozenDuringRun = document.getElementById("binder").disabled;
          pauseHeld = {
            label: document.getElementById("pause").textContent,
            stayed: held === stepText(),
            at: held,
            // 🔴 BOTH SAMPLES MUST FIND THE RUN STILL IN FLIGHT, or the
            // counter standing still says only that the run ENDED. That is how
            // this arm passed with the pause gate deleted.
            active: activeAt && running(),
          };
          document.getElementById("pause").click();
        }

        for (let i = 0; i < 240; i += 1) {
          await wait(500);
          if (!document.getElementById("stop").disabled) continue;
          if (document.getElementById("log").textContent.includes("ms/step")) break;
        }
        await wait(500);

        // 🔴 SWITCHING THE VIEW IS A HANDLER, AND A HANDLER IS WHERE THE BUGS
        // ARE. The dropdowns were added with no listener at all once already -
        // the page offered a control that did nothing. Dispatching the change
        //
        // 🔴 AND NO BACKTICK MAY APPEAR IN HERE, COMMENT OR NOT. Everything in
        // this block is inside a JS template literal, so a backtick ends the
        // string and the file stops parsing at a word inside a comment. Third
        // time in this repository; the other two were in WGSL.
        // event is the only way to see that the viewer's visible set actually
        // follows it, and that colouring by body does not throw.
        const viewSwitch = [];
        const preset = document.getElementById("view-preset");
        const colour = document.getElementById("colour-by");
        for (const id of ["everything", "backbone", "solute"]) {
          if (![...preset.options].some((o) => o.value === id)) continue;
          preset.value = id;
          preset.dispatchEvent(new Event("change"));
          await wait(200);
          viewSwitch.push({
            preset: id,
            visible: window.rainmd.viewer?.visibleCount ?? -1,
            counts: document.getElementById("view-counts").textContent,
          });
        }
        // 🔴 AND THE BINDER SPLIT IS A CONTROL TOO, with a consequence no other
        // dropdown has: it changes what every RMSD column MEANS. Driving it
        // checks that choosing a molecule re-splits the solute and that the
        // control is frozen while a run is in flight.
        const binderSwitch = [];
        const binder = document.getElementById("binder");
        if (binder !== null && !document.getElementById("binder-row").hidden) {
          const options = [...binder.options].map((o) => o.value);
          for (const value of options) {
            binder.value = value;
            binder.dispatchEvent(new Event("change"));
            await wait(150);
            const selections = window.rainmd.selections;
            binderSwitch.push({
              choice: value,
              receptor: selections?.receptor?.length ?? -1,
              binderAtoms: selections?.binder?.length ?? -1,
            });
          }
          binder.value = options[0];
          binder.dispatchEvent(new Event("change"));
          await wait(150);
        }

        // Every colour mode the page offers, each drawn under an error scope:
        // a mode that throws, or that validates but paints nothing, is a mode
        // the reader finds rather than the harness.
        const colourModes = [];
        if (colour !== null) {
          for (const mode of [...colour.options].map((o) => o.value)) {
            let error = null;
            try {
              colour.value = mode;
              colour.dispatchEvent(new Event("change"));
              await wait(350);
              // 🔴 WHAT 3Dmol PARSED, NOT A DEVICE ERROR SCOPE. The viewer owns
              // its own WebGL context now, so the simulation's WebGPU device
              // can say nothing about whether the structure drew. The atom
              // count is the checkable thing - and it is where the one bug that
              // matters hides, a parser silently dropping every hydrogen.
              const atoms = window.rainmd.viewer?.atomCount ?? 0;
              if (atoms === 0) error = "3Dmol parsed no atoms";
            } catch (thrown) {
              error = String(thrown && thrown.message || thrown);
            }
            colourModes.push({ mode, error, atoms: window.rainmd.viewer?.atomCount ?? 0 });
          }
        }
        const colourError = colourModes.find((m) => m.error !== null)?.error ?? null;

        // 🔴 AND EVERY REPRESENTATION, DRAWN AFTER A REAL RUN. The ribbon is
        // rebuilt from the frame on screen at every report, so the path that
        // matters is the one where the positions are moving - a ribbon that
        // builds correctly from a loaded file can still fail on the second
        // frame, when the buffers are reused rather than created.
        const drawModes = [];
        const representation = document.getElementById("representation");
        if (representation !== null) {
          for (const mode of [...representation.options].map((o) => o.value)) {
            let error = null;
            try {
              representation.value = mode;
              representation.dispatchEvent(new Event("change"));
              await wait(400);
              const atoms = window.rainmd.viewer?.atomCount ?? 0;
              if (atoms === 0) error = "3Dmol parsed no atoms";
            } catch (thrown) {
              error = String(thrown && thrown.message || thrown);
            }
            drawModes.push({ mode, error, atoms: window.rainmd.viewer?.atomCount ?? 0 });
          }
        }

        // Did the render pass validate? A broken pipeline or a depth-texture
        // size mismatch shows up here and nowhere in the DOM.
        let renderError = null;
        const handle = window.rainmd;
        if (handle.viewer !== null && (handle.viewer?.atomCount ?? 0) === 0) {
          renderError = "the viewer holds no atoms";
        }
        const analysis = handle.analysis;
        const last = analysis?.series?.[analysis.series.length - 1] ?? null;
        return JSON.stringify({
          log: document.getElementById("log").textContent,
          live: [...document.querySelectorAll("#live > div")].map(
            (cell) => cell.textContent),
          dcdEnabled: !document.getElementById("download-dcd").disabled,
          stateEnabled: !document.getElementById("download-state").disabled,
          pdbEnabled: !document.getElementById("download-pdb").disabled,
          metricsEnabled: !document.getElementById("download-metrics").disabled,
          rmsfEnabled: !document.getElementById("download-rmsf").disabled,
          viewerAtoms: handle.viewer?.atomCount ?? 0,
          viewerVisible: handle.viewer?.visibleCount ?? 0,
          presets: [...document.getElementById("view-preset").options].map((o) => o.textContent),
          counts: handle.selections?.counts ?? null,
          hasBinder: analysis?.hasBinder ?? false,
          frames: analysis?.series?.length ?? 0,
          rmsfFrames: analysis?.rmsf?.frames ?? 0,
          lastRow: last,
          firstRow: analysis?.series?.[0] ?? null,
          pauseHeld,
          // Production must be locked until equilibration has run, and
          // equilibration until minimisation has.
          equilibrateWasLockedBeforeMinimise: equilibrateDisabled,
          produceLockedAfterMinimise: document.getElementById("produce").disabled,
          restrainedAtoms: handle.lastRestrainedAtoms ?? null,
          viewSwitch,
          binderSwitch,
          binderFrozenDuringRun,
          colourError,
          colourModes,
          drawModes,
          // The files the export buttons would write, measured rather than
          // assumed: an enabled button says nothing about its contents.
          metricsLines: analysis === null ? 0 : analysis.metricsCsv().trim().split("\\n").length,
          rmsfLines: analysis === null ? 0 : analysis.rmsfCsv().trim().split("\\n").length,
          trajectoryBytes: handle.trajectoryBytes,
          renderError,
        });
      })()`;
    const result = await cdp(socketUrl, "Runtime.evaluate", {
      expression: simulateExpression.replace("FIXTURE_NAME", fixtureName),
      awaitPromise: true, returnByValue: true,
    });
    const payload = JSON.parse(cdpValue(result, "--simulate"));
    console.log("\n--- page log ---");
    console.log(payload.log.trim());
    console.log("\n--- live readout ---");
    for (const cell of payload.live) console.log(`  ${cell}`);
    console.log(`\nselection presets offered:`);
    for (const preset of payload.presets) console.log(`  ${preset}`);
    console.log(`\nlive analysis: ${payload.frames} frames, binder split: ${payload.hasBinder}, `
      + `RMSF over ${payload.rmsfFrames} frames`);
    if (payload.lastRow !== null) {
      const show = (key, digits = 4) => {
        const value = payload.lastRow[key];
        return value === null || value === undefined ? "–" : value.toFixed(digits);
      };
      console.log(`  RMSD complex ${show("rmsd")}  receptor ${show("rmsdReceptor")}  `
        + `binder ${show("rmsdBinder")}  binder-on-receptor ${show("rmsdBinderOnReceptor")} nm`);
      console.log(`  Rg ${show("radiusOfGyration")} nm   interface ${show("interfaceDistance")} nm`);
    }
    if (payload.firstRow !== null) {
      const f = payload.firstRow;
      console.log(`  frame 0: step ${f.step} T ${f.temperature?.toFixed(2)} K  `
        + `kinetic ${f.kinetic?.toFixed(2)}  potential ${f.potential?.toFixed(2)} kJ/mol`);
    }
    if (payload.pauseHeld === null) {
      console.log("\npause:                       not exercised (run finished first)");
    } else {
      const held = payload.pauseHeld;
      console.log(`\npause:                       button reads "${held.label}", `
        + `the integrator ${held.stayed ? "held at" : "ADVANCED PAST"} ${held.at}`
        + `, run still in flight: ${held.active}`);
      if (!held.stayed || !held.active || held.label !== "Resume") process.exitCode = 1;
    }
    console.log("\nview switching (the dropdown's own change handler):");
    for (const step of payload.viewSwitch) {
      console.log(`  ${step.preset.padEnd(12)} ${String(step.visible).padStart(5)} `
        + `visible   ${step.counts}`);
    }
    {
      const seen = new Set(payload.viewSwitch.map((v) => v.visible));
      // A handler that fires but changes nothing is the bug this arm exists for.
      if (payload.viewSwitch.length > 1 && seen.size < 2) process.exitCode = 1;
      if (payload.viewSwitch.some((v) => v.visible <= 0)) process.exitCode = 1;
    }
    if (payload.binderSwitch.length > 0) {
      console.log("\nbinder split (the molecule list a prmtop has instead of chains):");
      for (const step of payload.binderSwitch) {
        console.log(`  ${step.choice.padEnd(6)} receptor ${String(step.receptor).padStart(5)}`
          + ` / binder ${String(step.binderAtoms).padStart(5)}`);
      }
      const shapes = new Set(payload.binderSwitch.map((s) => `${s.receptor}/${s.binderAtoms}`));
      if (payload.binderSwitch.length > 1 && shapes.size < 2) process.exitCode = 1;
    }
    console.log(`binder frozen mid-run:       ${payload.binderFrozenDuringRun}`);
    console.log(`stage gating:                equilibrate locked before `
      + `minimise: ${payload.equilibrateWasLockedBeforeMinimise}`);
    if (payload.equilibrateWasLockedBeforeMinimise !== false) {
      // After the minimise click above it must be UNLOCKED; `true` here means
      // minimisation did not unlock it.
      process.exitCode = 1;
    }
    if (payload.binderFrozenDuringRun !== true) process.exitCode = 1;
    console.log(`draw modes:                  `
      + payload.drawModes.map((m) => `${m.mode}(${m.atoms} atoms)`
        + `${m.error === null ? "" : " FAILED"}`).join(", "));
    for (const mode of payload.drawModes) {
      if (mode.error !== null) {
        console.log(`  ${mode.mode}: ${mode.error}`);
        process.exitCode = 1;
      }
      // A ribbon mode that builds no segments is a cartoon nobody can see.
      if (mode.atoms === 0) process.exitCode = 1;
    }
    console.log(`colour modes:                `
      + payload.colourModes.map((m) => `${m.mode}${m.error === null ? " ok" : " FAILED"}`)
        .join(", "));
    if (payload.colourError !== null) {
      console.log(`  ${payload.colourError}`);
      process.exitCode = 1;
    }
    // One header plus one line per frame, and one row per residue.
    console.log(`exports:                     metrics.csv ${payload.metricsLines} lines `
      + `(${payload.frames} frames), rmsf.csv ${payload.rmsfLines} lines`);
    if (payload.metricsLines !== payload.frames + 1) process.exitCode = 1;
    console.log(`\nviewer atoms:                ${payload.viewerAtoms} `
      + `(${payload.viewerVisible} drawn)`);
    console.log(`render pass validated:       ${payload.renderError === null ? "yes" : payload.renderError}`);
    console.log(`downloads: dcd ${payload.dcdEnabled} (${payload.trajectoryBytes} b), `
      + `rst7 ${payload.stateEnabled}, pdb ${payload.pdbEnabled}, `
      + `metrics ${payload.metricsEnabled}, rmsf ${payload.rmsfEnabled}`);
    if (!payload.dcdEnabled || !payload.stateEnabled || !payload.pdbEnabled
      || !payload.metricsEnabled || !payload.rmsfEnabled) process.exitCode = 1;
    if (payload.frames < 2) process.exitCode = 1;
    if (payload.renderError !== null || payload.viewerAtoms === 0) process.exitCode = 1;
    if (/error:/i.test(payload.log)) process.exitCode = 1;

    // 🔴 A SCREENSHOT IS THE ONLY CHECK THAT THE VIEWER DREW ANYTHING. A render
    // pass can validate perfectly and produce an empty frame - a camera inside
    // the structure, a zero radius everywhere, a depth test that rejects every
    // fragment. Nothing in the API reports that. The PNG's size is a crude
    // proxy (a flat image compresses to almost nothing) and the file itself is
    // there to be looked at.
    await screenshot(socketUrl);
  }

  // 🔴 A LOAD-ONLY LANE, BECAUSE THE STRUCTURE FIXTURES HAVE NO BOX. The helix
  // and strand are vacuum peptides built for the viewer and the assignment, so
  // the GPU engine refuses them (it implements PME, which needs a periodic
  // cell) and `--simulate` cannot reach them. This drops the files on the page,
  // switches the colour mode, and reads what the viewer actually painted.
  const viewArg = process.argv.find((argument) => argument.startsWith("--view="));
  if (viewArg !== undefined) {
    const name = viewArg.slice("--view=".length);
    const expression = `(async () => {
        const wait = (ms) => new Promise((r) => setTimeout(r, ms));
        const fixture = "FIXTURE_NAME";
        const [top, crd] = await Promise.all([
          fetch("test/fixtures/" + fixture + ".prmtop").then((r) => r.text()),
          fetch("test/fixtures/" + fixture + ".inpcrd").then((r) => r.text()),
        ]);
        const files = new DataTransfer();
        files.items.add(new File([top], fixture + ".prmtop"));
        files.items.add(new File([crd], fixture + ".inpcrd"));
        const input = document.getElementById("file-input");
        input.files = files.files;
        input.dispatchEvent(new Event("change"));
        await wait(1500);
        document.getElementById("colour-by").value = "structure";
        document.getElementById("colour-by").dispatchEvent(new Event("change"));
        await wait(400);
        document.getElementById("representation").value = "DRAW_MODE";
        document.getElementById("representation").dispatchEvent(new Event("change"));
        await wait(600);
        let renderError = null;
        const handle = window.rainmd;
        // 🔴 THE CAMERA IS PART OF WHAT IS UNDER TEST. A flat ribbon seen
        // edge-on is a thin line, which is indistinguishable in a screenshot
        // from a ribbon that failed to build - the strand fixture lands exactly
        // that way on the default camera, because its broad face lies in the
        // sheet plane and the sheet plane faces the lens.
        // 🔴 NO WEBGPU ERROR SCOPE HERE ANY MORE. 3Dmol owns a WebGL context
        // of its own, so a device error scope on the simulation's WebGPU device
        // can say nothing about whether the structure drew. What IS checkable
        // is the model: how many atoms it parsed and how many chains it found,
        // which is also where the one bug that matters hides - a parser that
        // silently drops every hydrogen.
        if ("AZIMUTH" !== "") {
          try {
            handle.viewer?.viewer?.rotate?.(Number("AZIMUTH") * 57.2958, "y");
            handle.viewer?.viewer?.rotate?.(Number("ELEVATION") * 57.2958, "x");
            handle.viewer?.viewer?.render?.();
          } catch (error) { renderError = String(error.message ?? error); }
        }
        return JSON.stringify({
          log: document.getElementById("log").textContent,
          visible: handle.viewer?.visibleCount ?? 0,
          chains: handle.viewer?.chains?.length ?? 0,
          viewerAtoms: handle.viewer?.atomCount ?? 0,
          renderError,
        });
      })()`;
    const result = await cdp(socketUrl, "Runtime.evaluate", {
      // 🔴 replaceAll, NOT replace. Each placeholder appears twice - once in a
      // guard and once in the value - and `String.replace` with a string
      // pattern substitutes only the FIRST. The second AZIMUTH stayed literal,
      // `Number("AZIMUTH")` was NaN, the camera position was NaN, and the
      // render validated while drawing absolutely nothing. An empty screenshot
      // reads as a renderer that failed; it was the harness.
      expression: expression
        .replaceAll("FIXTURE_NAME", name)
        .replaceAll("DRAW_MODE", (process.argv.find((a) => a.startsWith("--draw="))
          ?? "--draw=ribbon").slice("--draw=".length))
        .replaceAll("AZIMUTH", (process.argv.find((a) => a.startsWith("--azimuth="))
          ?? "--azimuth=0.6").slice("--azimuth=".length))
        .replaceAll("ELEVATION", (process.argv.find((a) => a.startsWith("--elevation="))
          ?? "--elevation=0.35").slice("--elevation=".length)),
      awaitPromise: true, returnByValue: true,
    });
    const payload = JSON.parse(cdpValue(result, `--view=${name}`));
    console.log(`\n--- ${name} ---`);
    console.log(payload.log.trim());
    console.log(`viewer: ${payload.viewerAtoms} atoms parsed by 3Dmol, `
      + `${payload.chains} chain(s)`);
    if (payload.viewerAtoms === 0) process.exitCode = 1;
    console.log(`render:  ${payload.renderError === null ? "validated" : payload.renderError}`);
    if (payload.renderError !== null) process.exitCode = 1;
    if (!payload.log.includes("structure:")) process.exitCode = 1;
    if (payload.visible === 0) process.exitCode = 1;
    await screenshot(socketUrl);
  }

  // 🔴 THE EXAMPLE BUTTON IS THE ONLY WAY A VISITOR WITH NO AMBER FILES CAN
  // USE THE PAGE AT ALL, so it gets its own lane. It also checks the one thing
  // that cannot be seen on a dev server rooted at the repository: the fetches
  // are RELATIVE, so they survive being served from a subdirectory the way a
  // GitHub Pages project site is.
  if (process.argv.includes("--examples")) {
    const expression = `(async () => {
        const wait = (ms) => new Promise((r) => setTimeout(r, ms));
        const select = document.getElementById("example");
        const results = [];
        for (const name of [...select.options].map((o) => o.value)) {
          select.value = name;
          document.getElementById("load-example").click();
          let loaded = false;
          for (let i = 0; i < 60; i += 1) {
            await wait(250);
            if (!document.getElementById("load-example").disabled) { loaded = true; break; }
          }
          const error = document.getElementById("load-error");
          results.push({
            name,
            finished: loaded,
            atoms: window.rainmd.selections?.groups?.all?.length ?? 0,
            error: error.hidden ? null : error.textContent,
            log: document.getElementById("log").textContent.split("\\n")[0],
          });
        }
        return JSON.stringify({ results });
      })()`;
    const evaluation = await cdp(socketUrl, "Runtime.evaluate", {
      expression, awaitPromise: true, returnByValue: true,
    });
    const payload = JSON.parse(cdpValue(evaluation, "--examples"));
    console.log("\nbundled examples, loaded through the page's own button:");
    for (const row of payload.results) {
      console.log(`  ${row.name.padEnd(10)} ${String(row.atoms).padStart(5)} atoms  `
        + `${row.error === null ? row.log : `FAILED: ${row.error}`}`);
      if (row.error !== null || row.atoms === 0 || !row.finished) process.exitCode = 1;
    }
  }

  if (process.argv.includes("--probe-3dmol")) {
    const expression = `(async () => {
        const wait = (ms) => new Promise((r) => setTimeout(r, ms));
        const [top, crd] = await Promise.all([
          fetch("test/fixtures/mixed.prmtop").then((r) => r.text()),
          fetch("test/fixtures/mixed.inpcrd").then((r) => r.text()),
        ]);
        const files = new DataTransfer();
        files.items.add(new File([top], "mixed.prmtop"));
        files.items.add(new File([crd], "mixed.inpcrd"));
        const input = document.getElementById("file-input");
        input.files = files.files;
        input.dispatchEvent(new Event("change"));
        await wait(1500);
        const model = window.rainmd.viewer.model;
        const atoms = model.selectedAtoms({});
        const ss = {};
        for (const atom of atoms) { ss[atom.ss ?? "none"] = (ss[atom.ss ?? "none"] ?? 0) + 1; }
        return JSON.stringify({
          schemes: Object.keys(window.$3Dmol.builtinColorSchemes ?? {}),
          ssColors: Object.keys(window.$3Dmol.ssColors ?? {}),
          secondaryStructure: ss,
          sampleAtom: { resn: atoms[20]?.resn, ss: atoms[20]?.ss, chain: atoms[20]?.chain },
        });
      })()`;
    const evaluation = await cdp(socketUrl, "Runtime.evaluate", {
      expression, awaitPromise: true, returnByValue: true,
    });
    console.log(cdpValue(evaluation, "--probe-3dmol"));
  }

  // A whole-page screenshot at a chosen width, for looking at the layout.
  if (process.argv.includes("--layout")) {
    const expression = `(async () => {
        const wait = (ms) => new Promise((r) => setTimeout(r, ms));
        const [top, crd] = await Promise.all([
          fetch("test/fixtures/two-body.prmtop").then((r) => r.text()),
          fetch("test/fixtures/two-body.inpcrd").then((r) => r.text()),
        ]);
        const files = new DataTransfer();
        files.items.add(new File([top], "two-body.prmtop"));
        files.items.add(new File([crd], "two-body.inpcrd"));
        const input = document.getElementById("file-input");
        input.files = files.files;
        input.dispatchEvent(new Event("change"));
        await wait(2000);
        window.scrollTo(0, 0);
        const panel = document.getElementById("run-panel").getBoundingClientRect();
        const viewer = document.getElementById("viewer").getBoundingClientRect();
        return JSON.stringify({
          pageWidth: document.documentElement.clientWidth,
          viewerTop: Math.round(viewer.top + window.scrollY),
          viewerWidth: Math.round(viewer.width),
          panelWidth: Math.round(panel.width),
          horizontalScroll: document.documentElement.scrollWidth
            > document.documentElement.clientWidth + 1,
        });
      })()`;
    const evaluation = await cdp(socketUrl, "Runtime.evaluate", {
      expression, awaitPromise: true, returnByValue: true,
    });
    const payload = JSON.parse(cdpValue(evaluation, "--layout"));
    console.log(`layout at ${payload.pageWidth}px: viewer is ${payload.viewerWidth}px wide, `
      + `its top edge ${payload.viewerTop}px down the page`);
    // 🔴 THE NUMBER THAT WAS THE COMPLAINT. With everything in one column the
    // structure started roughly a screen down; it should now be near the top.
    if (payload.horizontalScroll) {
      console.log("  FAILED: the page scrolls horizontally");
      process.exitCode = 1;
    }
    await screenshot(socketUrl);
  }

  const moduleArg = process.argv.find((argument) => argument.startsWith("--module="));
  if (moduleArg !== undefined) {
    // 🔴 THE SAME CHECK MODULE AS THE NODE LANE, IMPORTED BY THE PAGE. Its
    // `main(device)` is lane-agnostic; only tools/gpu/fixture.js differs,
    // switching between readFileSync and fetch. That is what makes "it passes
    // in Chrome too" a statement about the same code rather than about a
    // second copy of it.
    const path = moduleArg.slice("--module=".length);
    const evaluation = await cdp(socketUrl, "Runtime.evaluate", {
      expression: `(async () => {
        try {
          const { requestMdDevice } = await import("/webgpu/runtime/device.js");
          const { device } = await requestMdDevice(navigator.gpu);
          const module = await import("/${path}");
          const result = await module.main(device);
          return JSON.stringify(result);
        } catch (error) {
          return JSON.stringify({ error: String(error && error.stack || error) });
        }
      })()`,
      awaitPromise: true, returnByValue: true,
    });
    const payload = JSON.parse(cdpValue(evaluation, `--module=${path}`));
    if (payload.error !== undefined) {
      console.error(`\nin-page failure:\n${payload.error}`);
      process.exitCode = 1;
    } else {
      console.log(`\n--- ${path} in Chrome ---`);
      console.log(JSON.stringify(payload, null, 1));
      if (payload.failures > 0) process.exitCode = 1;
    }
  }

  if (runCheck) {
    const clicked = await cdp(socketUrl, "Runtime.evaluate", {
      expression: `(async () => {
        const input = document.getElementById("file-input");
        const fixture = "FIXTURE_NAME";
        const [top, crd] = await Promise.all([
          fetch("test/fixtures/" + fixture + ".prmtop").then((r) => r.text()),
          fetch("test/fixtures/" + fixture + ".inpcrd").then((r) => r.text()),
        ]);
        const files = new DataTransfer();
        files.items.add(new File([top], fixture + ".prmtop"));
        files.items.add(new File([crd], fixture + ".inpcrd"));
        input.files = files.files;
        input.dispatchEvent(new Event("change"));
        await new Promise((r) => setTimeout(r, 1500));
        document.getElementById("check-gpu").click();
        await new Promise((r) => setTimeout(r, 6000));
        return document.getElementById("log").textContent;
      })()`,
      awaitPromise: true, returnByValue: true,
    });
    console.log("\n--- page log ---");
    console.log(clicked.result?.result?.value ?? "(no log)");
  }
} finally {
  chrome.kill();
  server.kill();
}
