// Small DOM helpers shared by the four panels.
//
// Deliberately framework-free: thin wrappers over document.createElement
// plus a couple of widgets used across panels (command display with a copy
// button, status banners, JSON/report rendering).

import { ApiError } from "./api.js";

/** Create an element with optional attributes/props and children. */
export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  props: Partial<HTMLElementTagNameMap[K]> & { class?: string } = {},
  children: (Node | string)[] = [],
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") {
      node.className = v as string;
    } else if (v !== undefined) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (node as any)[k] = v;
    }
  }
  for (const c of children) {
    node.append(c);
  }
  return node;
}

/** Find a required element by id; throw if absent (fail fast at startup). */
export function byId<T extends HTMLElement = HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) {
    throw new Error(`missing element #${id}`);
  }
  return node as T;
}

/**
 * Render the CLI `command` string into a container with a copy button.
 *
 * The CLI is the source of truth, so every API result shows its command.
 */
export function showCommand(container: HTMLElement, command: string): void {
  container.replaceChildren();
  if (!command) {
    return;
  }
  const code = el("code", { class: "cmd-text", textContent: command });
  const btn = el("button", { class: "copy-btn", type: "button", textContent: "Copy" });
  btn.addEventListener("click", () => {
    void navigator.clipboard.writeText(command).then(
      () => {
        const prev = btn.textContent;
        btn.textContent = "Copied";
        window.setTimeout(() => {
          btn.textContent = prev;
        }, 1200);
      },
      () => {
        btn.textContent = "Copy failed";
      },
    );
  });
  const label = el("span", { class: "cmd-label", textContent: "Equivalent CLI:" });
  container.append(el("div", { class: "cmd-box" }, [label, code, btn]));
}

/** Status banner level. */
export type Level = "info" | "ok" | "warn" | "error";

/** Render a one-line status banner into a container. */
export function showStatus(container: HTMLElement, level: Level, message: string): void {
  container.replaceChildren(el("div", { class: `status status-${level}`, textContent: message }));
}

/** Clear a container. */
export function clear(container: HTMLElement): void {
  container.replaceChildren();
}

/** Render an ApiError (or any error) into a status container. */
export function showError(container: HTMLElement, e: unknown): void {
  let msg: string;
  if (e instanceof ApiError) {
    msg = e.detail ? `${e.message}: ${e.detail}` : e.message;
  } else if (e instanceof Error) {
    msg = e.message;
  } else {
    msg = String(e);
  }
  showStatus(container, "error", msg);
}

/** Render an arbitrary value (report payload) as pretty JSON in a <pre>. */
export function showJson(container: HTMLElement, value: unknown): void {
  let text: string;
  if (typeof value === "string") {
    text = value;
  } else {
    try {
      text = JSON.stringify(value, null, 2);
    } catch {
      text = String(value);
    }
  }
  container.replaceChildren(el("pre", { class: "report", textContent: text }));
}

/**
 * Render a list section (e.g. errors/warnings/info) with a heading.
 * Appends nothing if the list is empty.
 */
export function appendMessageList(
  container: HTMLElement,
  title: string,
  level: Level,
  items: string[] | undefined,
): void {
  if (!items || items.length === 0) {
    return;
  }
  const heading = el("h4", { class: `list-head list-${level}`, textContent: `${title} (${items.length})` });
  const ul = el(
    "ul",
    { class: `msg-list msg-${level}` },
    items.map((it) => el("li", { textContent: it })),
  );
  container.append(heading, ul);
}

/**
 * Render an array of loosely-typed rows as a table. Columns are the union
 * of keys across rows (ordered by first appearance). Used for the quality
 * report's per-feature / per-video tables.
 */
export function renderTable(rows: Record<string, unknown>[]): HTMLElement | null {
  if (!rows || rows.length === 0) {
    return null;
  }
  const cols: string[] = [];
  for (const row of rows) {
    for (const k of Object.keys(row)) {
      if (!cols.includes(k)) {
        cols.push(k);
      }
    }
  }
  const thead = el("thead", {}, [
    el(
      "tr",
      {},
      cols.map((c) => el("th", { textContent: c })),
    ),
  ]);
  const tbody = el(
    "tbody",
    {},
    rows.map((row) =>
      el(
        "tr",
        {},
        cols.map((c) => {
          const v = row[c];
          const text = v === null || v === undefined ? "" : typeof v === "object" ? JSON.stringify(v) : String(v);
          return el("td", { textContent: text });
        }),
      ),
    ),
  );
  return el("table", { class: "data-table" }, [thead, tbody]);
}
