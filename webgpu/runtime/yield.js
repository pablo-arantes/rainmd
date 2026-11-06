/**
 * Hand control back to the browser without being throttled for it.
 *
 * 🔴 setTimeout IS CLAMPED TO >= 1 SECOND IN A BACKGROUND TAB, AND AN MD RUN
 * IS THE WORST POSSIBLE VICTIM OF THAT. A simulation yields between step
 * blocks - hundreds of times a second - so a run that takes ten minutes in a
 * visible tab takes hours in a hidden one, and the user's conclusion is that
 * the page stopped. The GPU does not care whether anyone is looking; only the
 * yield does.
 *
 * A MessageChannel message is a task, not a timer, so it is not clamped, and it
 * still lets the browser paint and still lets a Stop button land.
 *
 * 🔴 AND IT MUST BE A REAL TASK, NOT A RESOLVED PROMISE. Awaiting an
 * already-resolved promise is a microtask: control returns to the microtask
 * queue and never to the browser, so the page does not repaint and the Stop
 * button does not respond. That is the bug this replaces, not one it adds.
 */
let channel;

export function yieldToBrowser() {
  if (typeof MessageChannel === "undefined") {
    return new Promise((resolve) => { setTimeout(resolve, 0); });
  }
  if (channel === undefined) channel = new MessageChannel();
  return new Promise((resolve) => {
    channel.port1.onmessage = () => resolve();
    channel.port2.postMessage(undefined);
  });
}

/**
 * A cooperative stop flag an encode loop can check between submits.
 *
 * MD has no natural place to abort: a step is a handful of dispatches and
 * cancelling mid-step leaves positions updated and velocities not. So the
 * signal is only ever read at a step boundary, and a stopped run is a complete
 * state that can be written out and resumed.
 */
export class StopSignal {
  constructor() { this._stopped = false; this._reason = null; }
  stop(reason = "stopped by the user") { this._stopped = true; this._reason = reason; }
  get stopped() { return this._stopped; }
  throwIfStopped() {
    if (this._stopped) {
      const error = new Error(this._reason);
      error.name = "SimulationStopped";
      throw error;
    }
  }
}
