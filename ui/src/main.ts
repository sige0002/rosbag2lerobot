// Tab wiring for the bagel UI.
//
// One page, four tabs (one section visible at a time) mirroring the
// convert loop:
//   1. Bags    -> browse + inspect (per-bag table)
//   2. Config  -> template / scaffold / options -> validate-config
//   3. Convert -> poll progress
//   4. Results -> quality-report / validate-dataset / preview iframe
//
// State (selected bags, config YAML, output dir, dataset) is shared across
// tabs. Each action displays the CLI `command` returned by its API call.

import * as api from "./api.js";
import type { BrowseEntry, InspectBag, InspectTopic, Root } from "./api.js";
import {
  appendMessageList,
  byId,
  clear,
  el,
  renderTable,
  showCommand,
  showError,
  showJson,
  showStatus,
} from "./panels.js";

// ---------------------------------------------------------------------------
// Tab navigation.
// ---------------------------------------------------------------------------

type TabId = "bags" | "config" | "convert" | "results";

/** A tab's nav button + its content section. */
interface Tab {
  btn: HTMLButtonElement;
  section: HTMLElement;
  /** Called once each time the tab is shown (e.g. lazy-load lists). */
  onShow?: () => void;
}

const tabs: Record<TabId, Tab> = {
  bags: { btn: byId<HTMLButtonElement>("tab-bags"), section: byId("panel-browse") },
  config: { btn: byId<HTMLButtonElement>("tab-config"), section: byId("panel-config") },
  convert: { btn: byId<HTMLButtonElement>("tab-convert"), section: byId("panel-convert") },
  results: { btn: byId<HTMLButtonElement>("tab-results"), section: byId("panel-quality") },
};

function selectTab(id: TabId): void {
  for (const [key, tab] of Object.entries(tabs) as [TabId, Tab][]) {
    const active = key === id;
    tab.btn.setAttribute("aria-selected", active ? "true" : "false");
    tab.section.hidden = !active;
  }
  tabs[id].onShow?.();
}

for (const [id, tab] of Object.entries(tabs) as [TabId, Tab][]) {
  tab.btn.addEventListener("click", () => selectTab(id));
}

// ---------------------------------------------------------------------------
// Shared cross-panel state.
// ---------------------------------------------------------------------------

interface AppState {
  rootId: string | null;
  subpath: string;
  /** Relative bag paths the user has selected (checkbox state). */
  selectedBags: Set<string>;
  /** Output dir produced by the last convert (for quality/preview panel). */
  lastOutput: string | null;
}

const state: AppState = {
  rootId: null,
  subpath: "",
  selectedBags: new Set<string>(),
  lastOutput: null,
};

/** Selected bags as a sorted array (stable order for API calls). */
function selectedBagList(): string[] {
  return [...state.selectedBags].sort();
}

/** Disable a button while an async action runs; restore afterwards. */
async function withBusy<T>(btn: HTMLButtonElement, fn: () => Promise<T>): Promise<T | undefined> {
  const wasDisabled = btn.disabled;
  btn.disabled = true;
  try {
    return await fn();
  } finally {
    btn.disabled = wasDisabled;
  }
}

// ===========================================================================
// Panel 1: Browse bag
// ===========================================================================

const rootSelect = byId<HTMLSelectElement>("root-select");
const pathLabel = byId("browse-path");
const upBtn = byId<HTMLButtonElement>("browse-up");
const entriesEl = byId("browse-entries");
const selectedEl = byId("browse-selected");
const inspectBtn = byId<HTMLButtonElement>("inspect-btn");
const inspectCmd = byId("inspect-cmd");
const inspectStatus = byId("inspect-status");
const inspectReport = byId("inspect-report");

async function loadConfig(): Promise<void> {
  try {
    const cfg = await api.getConfig();
    byId("bagel-version").textContent = cfg.bagel_version ? `bagel ${cfg.bagel_version}` : "";
    rootSelect.replaceChildren(
      ...cfg.roots.map((r: Root) => el("option", { value: r.id, textContent: `${r.label} (${r.path})` })),
    );
    if (cfg.roots.length > 0) {
      const first = cfg.roots[0];
      if (first) {
        state.rootId = first.id;
        rootSelect.value = first.id;
        await loadBrowse("");
      }
    } else {
      showStatus(inspectStatus, "warn", "No roots configured.");
    }
  } catch (e) {
    showError(inspectStatus, e);
  }
}

async function loadBrowse(subpath: string): Promise<void> {
  if (!state.rootId) {
    return;
  }
  try {
    const res = await api.browse(state.rootId, subpath);
    state.subpath = subpath;
    pathLabel.textContent = res.path;
    upBtn.disabled = subpath === "";
    renderEntries(res.entries);
  } catch (e) {
    showError(inspectStatus, e);
  }
}

/** Join the current subpath with a child name (POSIX-style, relative). */
function joinSub(name: string): string {
  return state.subpath ? `${state.subpath}/${name}` : name;
}

function renderEntries(entries: BrowseEntry[]): void {
  clear(entriesEl);
  if (entries.length === 0) {
    entriesEl.append(el("div", { class: "muted", textContent: "(empty)" }));
    return;
  }
  for (const entry of entries) {
    const row = el("div", { class: "entry" });
    if (entry.is_bag) {
      const rel = joinSub(entry.name);
      const cb = el("input", { type: "checkbox", checked: state.selectedBags.has(rel) });
      cb.addEventListener("change", () => {
        if (cb.checked) {
          state.selectedBags.add(rel);
        } else {
          state.selectedBags.delete(rel);
        }
        renderSelected();
      });
      row.append(
        cb,
        el("span", { class: "entry-icon", textContent: "[bag]" }),
        el("span", { class: "entry-name", textContent: entry.name }),
      );
    } else if (entry.is_dir) {
      const link = el("button", { class: "dir-link", type: "button", textContent: entry.name });
      link.addEventListener("click", () => void loadBrowse(joinSub(entry.name)));
      row.append(el("span", { class: "entry-icon", textContent: "[dir]" }), link);
    } else {
      row.append(
        el("span", { class: "entry-icon", textContent: "     " }),
        el("span", { class: "entry-name muted", textContent: entry.name }),
      );
    }
    entriesEl.append(row);
  }
}

function renderSelected(): void {
  const bags = selectedBagList();
  clear(selectedEl);
  if (bags.length === 0) {
    selectedEl.append(el("span", { class: "muted", textContent: "No bags selected." }));
  } else {
    selectedEl.append(
      el("strong", { textContent: `${bags.length} bag(s):` }),
      ...bags.map((b) => {
        const chip = el("span", { class: "chip" });
        const x = el("button", { class: "chip-x", type: "button", textContent: "x", title: "remove" });
        x.addEventListener("click", () => {
          state.selectedBags.delete(b);
          renderSelected();
          // Re-render entries so checkbox state stays in sync if visible.
          void loadBrowse(state.subpath);
        });
        chip.append(el("span", { textContent: b }), x);
        return chip;
      }),
    );
  }
  inspectBtn.disabled = bags.length === 0;
  refreshHints();
}

rootSelect.addEventListener("change", () => {
  state.rootId = rootSelect.value;
  void loadBrowse("");
});

upBtn.addEventListener("click", () => {
  const idx = state.subpath.lastIndexOf("/");
  void loadBrowse(idx >= 0 ? state.subpath.slice(0, idx) : "");
});

inspectBtn.addEventListener("click", () => {
  void withBusy(inspectBtn, async () => {
    showStatus(inspectStatus, "info", "Inspecting...");
    clear(inspectReport);
    try {
      const res = await api.inspect(selectedBagList());
      showCommand(inspectCmd, res.command);
      clear(inspectStatus);
      renderInspect(res.report);
    } catch (e) {
      showError(inspectStatus, e);
    }
  });
});

/** Format a number to a fixed precision, blanking null/undefined. */
function fmt(value: number | null | undefined, digits = 1): string {
  return value == null ? "" : value.toFixed(digits);
}

/**
 * Render the inspect report as one table per bag: topic | msg_type | count |
 * mean fps. Falls back to raw JSON if the shape is unexpected.
 */
function renderInspect(report: { bags?: InspectBag[] }): void {
  clear(inspectReport);
  const bags = report?.bags;
  if (!Array.isArray(bags) || bags.length === 0) {
    showJson(inspectReport, report);
    return;
  }
  for (const bag of bags) {
    inspectReport.append(el("h4", { textContent: bag.bag }));
    const topics: InspectTopic[] = Array.isArray(bag.topics) ? bag.topics : [];
    const rows = topics.map((t) => ({
      topic: t.name,
      msg_type: t.msg_type,
      count: t.msg_count,
      "mean fps": fmt(t.fps?.mean),
    }));
    const table = renderTable(rows);
    if (table) {
      inspectReport.append(table);
    } else {
      inspectReport.append(el("div", { class: "muted", textContent: "(no topics)" }));
    }
  }
}

// ===========================================================================
// Panel 2: Config (template / scaffold / options / validate)
// ===========================================================================

const templateSelect = byId<HTMLSelectElement>("config-template-select");
const templateBtn = byId<HTMLButtonElement>("config-template-btn");
const templateCmd = byId("config-template-cmd");
const robotTypeInput = byId<HTMLInputElement>("scaffold-robot-type");
const taskInput = byId<HTMLInputElement>("scaffold-task");
const scaffoldFpsInput = byId<HTMLInputElement>("scaffold-fps");
const scaffoldBtn = byId<HTMLButtonElement>("scaffold-btn");
const scaffoldCmd = byId("scaffold-cmd");
const scaffoldStatus = byId("scaffold-status");
const configText = byId<HTMLTextAreaElement>("config-text");
const configHint = byId("config-hint");
// Option form controls.
const optRobotType = byId<HTMLInputElement>("opt-robot-type");
const optTask = byId<HTMLInputElement>("opt-task");
const optFps = byId<HTMLInputElement>("opt-fps");
const optPolicy = byId<HTMLSelectElement>("opt-policy");
const optStampSource = byId<HTMLSelectElement>("opt-stamp-source");
const optTolerance = byId<HTMLInputElement>("opt-tolerance");
const optAlign = byId<HTMLInputElement>("opt-align");
const configApplyBtn = byId<HTMLButtonElement>("config-apply-btn");
const configApplyCmd = byId("config-apply-cmd");
const configApplyStatus = byId("config-apply-status");
const validateConfigBtn = byId<HTMLButtonElement>("validate-config-btn");
const validateConfigCmd = byId("validate-config-cmd");
const validateConfigStatus = byId("validate-config-status");
const validateConfigReport = byId("validate-config-report");

function numOrUndef(input: HTMLInputElement): number | undefined {
  const v = input.value.trim();
  if (v === "") {
    return undefined;
  }
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}

function strOrUndef(input: HTMLInputElement): string | undefined {
  const v = input.value.trim();
  return v === "" ? undefined : v;
}

// --- Template picker --------------------------------------------------------

let templatesLoaded = false;

/** Lazily populate the template dropdown the first time the tab is shown. */
async function loadTemplates(): Promise<void> {
  if (templatesLoaded) {
    return;
  }
  try {
    const res = await api.configList();
    templateSelect.replaceChildren(
      ...res.configs.map((c) =>
        el("option", {
          value: c.name,
          textContent: c.robot_type ? `${c.name} (${c.robot_type})` : c.name,
        }),
      ),
    );
    templatesLoaded = true;
    templateBtn.disabled = res.configs.length === 0;
    if (res.configs.length === 0) {
      templateSelect.replaceChildren(el("option", { value: "", textContent: "(no templates)" }));
    }
  } catch (e) {
    showError(scaffoldStatus, e);
  }
}

templateBtn.addEventListener("click", () => {
  void withBusy(templateBtn, async () => {
    const name = templateSelect.value;
    if (!name) {
      showStatus(scaffoldStatus, "warn", "No template selected.");
      return;
    }
    showStatus(scaffoldStatus, "info", "Loading template...");
    try {
      const res = await api.configTemplate(name);
      showCommand(templateCmd, res.command);
      configText.value = res.yaml;
      clear(scaffoldStatus);
    } catch (e) {
      showError(scaffoldStatus, e);
    }
  });
});

// --- Scaffold from bag ------------------------------------------------------

scaffoldBtn.addEventListener("click", () => {
  void withBusy(scaffoldBtn, async () => {
    const bags = selectedBagList();
    if (bags.length === 0) {
      showStatus(scaffoldStatus, "warn", "Select at least one bag in the Bags tab first.");
      return;
    }
    showStatus(scaffoldStatus, "info", "Generating config...");
    try {
      const opts: api.ScaffoldOpts = {};
      const rt = strOrUndef(robotTypeInput);
      const tk = strOrUndef(taskInput);
      const fps = numOrUndef(scaffoldFpsInput);
      if (rt !== undefined) opts.robot_type = rt;
      if (tk !== undefined) opts.task = tk;
      if (fps !== undefined) opts.fps = fps;
      const res = await api.scaffold(bags, opts);
      showCommand(scaffoldCmd, res.command);
      configText.value = res.yaml;
      clear(scaffoldStatus);
    } catch (e) {
      showError(scaffoldStatus, e);
    }
  });
});

// --- Apply global options ---------------------------------------------------

/** Collect only the option fields the user actually set. */
function collectOptions(): api.ConfigApplyOptions {
  const opts: api.ConfigApplyOptions = {};
  const rt = strOrUndef(optRobotType);
  const tk = strOrUndef(optTask);
  const fps = numOrUndef(optFps);
  const tol = numOrUndef(optTolerance);
  if (rt !== undefined) opts.robot_type = rt;
  if (tk !== undefined) opts.task = tk;
  if (fps !== undefined) opts.fps = fps;
  if (optPolicy.value) opts.resampling_policy = optPolicy.value;
  if (optStampSource.value) opts.stamp_source = optStampSource.value;
  if (tol !== undefined) opts.tolerance_ms = tol;
  // The checkbox has no "unset" state; only send it when ticked.
  if (optAlign.checked) opts.align_to_required = true;
  return opts;
}

configApplyBtn.addEventListener("click", () => {
  void withBusy(configApplyBtn, async () => {
    const options = collectOptions();
    if (Object.keys(options).length === 0) {
      showStatus(configApplyStatus, "warn", "Set at least one option to apply.");
      return;
    }
    showStatus(configApplyStatus, "info", "Applying options...");
    try {
      const res = await api.configApply(configText.value, options);
      showCommand(configApplyCmd, res.command);
      configText.value = res.yaml;
      clear(configApplyStatus);
    } catch (e) {
      showError(configApplyStatus, e);
    }
  });
});

// --- Validate config --------------------------------------------------------

validateConfigBtn.addEventListener("click", () => {
  void withBusy(validateConfigBtn, async () => {
    const bags = selectedBagList();
    if (bags.length === 0) {
      showStatus(validateConfigStatus, "warn", "Select at least one bag in the Bags tab first.");
      return;
    }
    if (configText.value.trim() === "") {
      showStatus(validateConfigStatus, "warn", "Config is empty. Load, scaffold, or paste a config first.");
      return;
    }
    showStatus(validateConfigStatus, "info", "Validating config...");
    clear(validateConfigReport);
    try {
      const res = await api.validateConfig(configText.value, bags);
      showCommand(validateConfigCmd, res.command);
      renderValidation(validateConfigStatus, validateConfigReport, res.report);
    } catch (e) {
      showError(validateConfigStatus, e);
    }
  });
});

/** Shared rendering for validate-config / validate-dataset reports. */
function renderValidation(
  statusEl: HTMLElement,
  reportEl: HTMLElement,
  report: api.ValidationReport,
): void {
  const verdict = report.verdict ?? "(no verdict)";
  const errs = report.errors ?? [];
  const level = errs.length > 0 ? "error" : (report.warnings?.length ?? 0) > 0 ? "warn" : "ok";
  showStatus(statusEl, level, `Verdict: ${verdict}`);
  clear(reportEl);
  appendMessageList(reportEl, "Errors", "error", report.errors);
  appendMessageList(reportEl, "Warnings", "warn", report.warnings);
  appendMessageList(reportEl, "Info", "info", report.info);
}

// ===========================================================================
// Panel 3: Convert (progress)
// ===========================================================================

const outputInput = byId<HTMLInputElement>("convert-output");
const convertFpsInput = byId<HTMLInputElement>("convert-fps");
const convertWorkersInput = byId<HTMLInputElement>("convert-workers");
const convertCodecInput = byId<HTMLInputElement>("convert-codec");
const convertBtn = byId<HTMLButtonElement>("convert-btn");
const convertCmd = byId("convert-cmd");
const convertStatus = byId("convert-status");
const progressWrap = byId("convert-progress");
const progressBar = byId("convert-bar");
const progressText = byId("convert-progress-text");
const convertSummary = byId("convert-summary");

let pollTimer: number | null = null;

function stopPolling(): void {
  if (pollTimer !== null) {
    window.clearTimeout(pollTimer);
    pollTimer = null;
  }
}

function renderProgress(p: api.ConvertProgress): void {
  progressWrap.hidden = false;
  const pct = p.total > 0 ? Math.round((p.done / p.total) * 100) : 0;
  progressBar.style.width = `${pct}%`;
  const fail = p.n_failed > 0 ? `, ${p.n_failed} failed` : "";
  progressText.textContent = `${p.done} / ${p.total} (${pct}%)${fail}`;
}

async function pollJob(jobId: string): Promise<void> {
  try {
    const st = await api.convertStatus(jobId);
    renderProgress(st.progress);
    if (st.state === "running") {
      showStatus(convertStatus, "info", "Converting...");
      pollTimer = window.setTimeout(() => void pollJob(jobId), 1000);
      return;
    }
    stopPolling();
    if (st.state === "done") {
      const failed = st.progress.n_failed;
      if (failed > 0) {
        showStatus(convertStatus, "warn", `Done with ${failed} failure(s).`);
      } else {
        showStatus(convertStatus, "ok", "Conversion complete.");
      }
      // Make this output available to the Results tab.
      state.lastOutput = outputInput.value.trim();
      datasetInput.value = state.lastOutput;
      refreshHints();
      if (st.summary != null) {
        showJson(convertSummary, st.summary);
      }
    } else {
      showStatus(convertStatus, "error", st.error ?? "Conversion failed.");
      if (st.summary != null) {
        showJson(convertSummary, st.summary);
      }
    }
  } catch (e) {
    stopPolling();
    showError(convertStatus, e);
  } finally {
    if (pollTimer === null) {
      convertBtn.disabled = false;
    }
  }
}

convertBtn.addEventListener("click", () => {
  const bags = selectedBagList();
  if (bags.length === 0) {
    showStatus(convertStatus, "warn", "Select at least one bag in Panel 1 first.");
    return;
  }
  if (configText.value.trim() === "") {
    showStatus(convertStatus, "warn", "Config is empty (Panel 2).");
    return;
  }
  const output = outputInput.value.trim();
  if (output === "") {
    showStatus(convertStatus, "warn", "Enter an output directory (relative).");
    return;
  }
  stopPolling();
  clear(convertSummary);
  convertBtn.disabled = true;
  showStatus(convertStatus, "info", "Starting conversion...");
  void (async () => {
    try {
      const opts: api.ConvertOpts = {};
      const fps = numOrUndef(convertFpsInput);
      const workers = numOrUndef(convertWorkersInput);
      const codec = strOrUndef(convertCodecInput);
      if (fps !== undefined) opts.fps = fps;
      if (workers !== undefined) opts.workers = workers;
      if (codec !== undefined) opts.video_codec = codec;
      const res = await api.convert(configText.value, bags, output, opts);
      showCommand(convertCmd, res.command);
      renderProgress({ done: 0, total: 0, n_failed: 0 });
      void pollJob(res.job_id);
    } catch (e) {
      showError(convertStatus, e);
      convertBtn.disabled = false;
    }
  })();
});

// ===========================================================================
// Panel 4: Quality + preview (results)
// ===========================================================================

const datasetInput = byId<HTMLInputElement>("dataset-input");
const qualityBtn = byId<HTMLButtonElement>("quality-btn");
const qualityCmd = byId("quality-cmd");
const qualityStatus = byId("quality-status");
const qualityReportEl = byId("quality-report");
const validateDsBtn = byId<HTMLButtonElement>("validate-ds-btn");
const validateDsCmd = byId("validate-ds-cmd");
const validateDsStatus = byId("validate-ds-status");
const validateDsReport = byId("validate-ds-report");
const previewBtn = byId<HTMLButtonElement>("preview-btn");
const previewFrame = byId<HTMLIFrameElement>("preview-frame");

function currentDataset(): string {
  return datasetInput.value.trim() || (state.lastOutput ?? "");
}

qualityBtn.addEventListener("click", () => {
  void withBusy(qualityBtn, async () => {
    const ds = currentDataset();
    if (ds === "") {
      showStatus(qualityStatus, "warn", "Enter a dataset path (or run a conversion first).");
      return;
    }
    showStatus(qualityStatus, "info", "Building quality report...");
    clear(qualityReportEl);
    try {
      const res = await api.qualityReport(ds);
      showCommand(qualityCmd, res.command);
      renderQuality(res.report);
      clear(qualityStatus);
    } catch (e) {
      showError(qualityStatus, e);
    }
  });
});

function renderQuality(report: api.QualityReport): void {
  clear(qualityReportEl);
  const score = report.score ?? "n/a";
  const verdict = report.verdict ?? "(no verdict)";
  qualityReportEl.append(
    el("div", { class: "quality-head" }, [
      el("span", { class: "score", textContent: `Score: ${String(score)}` }),
      el("span", { class: "verdict", textContent: `Verdict: ${verdict}` }),
    ]),
  );
  if (report.features && report.features.length > 0) {
    const table = renderTable(report.features);
    if (table) {
      qualityReportEl.append(el("h4", { textContent: "Features" }), table);
    }
  }
  if (report.videos && report.videos.length > 0) {
    const table = renderTable(report.videos);
    if (table) {
      qualityReportEl.append(el("h4", { textContent: "Videos" }), table);
    }
  }
}

validateDsBtn.addEventListener("click", () => {
  void withBusy(validateDsBtn, async () => {
    const ds = currentDataset();
    if (ds === "") {
      showStatus(validateDsStatus, "warn", "Enter a dataset path (or run a conversion first).");
      return;
    }
    showStatus(validateDsStatus, "info", "Validating dataset...");
    clear(validateDsReport);
    try {
      const res = await api.validateDataset(ds);
      showCommand(validateDsCmd, res.command);
      renderValidation(validateDsStatus, validateDsReport, res.report);
    } catch (e) {
      showError(validateDsStatus, e);
    }
  });
});

previewBtn.addEventListener("click", () => {
  const ds = currentDataset();
  if (ds === "") {
    showStatus(qualityStatus, "warn", "Enter a dataset path (or run a conversion first).");
    return;
  }
  previewFrame.hidden = false;
  previewFrame.src = api.previewUrl(ds);
});

// ===========================================================================
// Cross-tab hints + lazy loading
// ===========================================================================

const convertHint = byId("convert-hint");
const resultsHint = byId("results-hint");

/**
 * Show/hide the per-tab prerequisite hints. Later tabs stay clickable; the
 * hint just tells the user what is missing.
 */
function refreshHints(): void {
  const hasBags = state.selectedBags.size > 0;
  const hasConfig = configText.value.trim() !== "";
  configHint.hidden = hasBags;
  convertHint.hidden = hasBags && hasConfig;
  resultsHint.hidden = currentDataset() !== "";
}

// Lazy-load the template list the first time the Config tab opens; always
// refresh hints when entering Config / Convert / Results.
tabs.config.onShow = () => {
  void loadTemplates();
  refreshHints();
};
tabs.convert.onShow = refreshHints;
tabs.results.onShow = refreshHints;

// Keep hints fresh as the user edits the config (affects Convert prereqs).
configText.addEventListener("input", refreshHints);

// ===========================================================================
// Startup
// ===========================================================================

renderSelected();
refreshHints();
void loadConfig();
