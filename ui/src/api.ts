// Typed fetch wrapper + interfaces matching the bagel UI HTTP contract.
//
// Base URL = current origin. The auth token is read once from
// `?token=<t>` in location.search and sent as the `X-Bagel-Token`
// header on every /api/* request. The preview route is GET and is
// loaded via an <iframe>, so it carries the token as a query param.

/** Auth token read once from the page URL (`?token=<t>`). May be "". */
export const TOKEN: string = new URLSearchParams(location.search).get("token") ?? "";

// ---------------------------------------------------------------------------
// Response/request interfaces (mirror the API contract exactly).
// ---------------------------------------------------------------------------

export interface Root {
  id: string;
  label: string;
  path: string;
}

export interface ConfigResponse {
  roots: Root[];
  bagel_version: string;
}

export interface BrowseEntry {
  name: string;
  is_dir: boolean;
  is_bag: boolean;
}

export interface BrowseResponse {
  path: string;
  entries: BrowseEntry[];
  bags: string[];
}

export interface InspectResponse {
  command: string;
  report: unknown;
}

export interface ScaffoldResponse {
  command: string;
  yaml: string;
}

/** A validation report: verdict + categorized message lists. */
export interface ValidationReport {
  verdict?: string;
  errors?: string[];
  warnings?: string[];
  info?: string[];
  [key: string]: unknown;
}

export interface ValidateConfigResponse {
  command: string;
  report: ValidationReport;
}

export interface ValidateDatasetResponse {
  command: string;
  report: ValidationReport;
}

export interface ConvertResponse {
  command: string;
  job_id: string;
}

export interface ConvertProgress {
  done: number;
  total: number;
  n_failed: number;
}

export interface ConvertStatus {
  state: "running" | "done" | "failed";
  progress: ConvertProgress;
  summary: unknown | null;
  error: string | null;
}

/** A per-feature or per-video row in the quality report. Shape is loose. */
export type QualityRow = Record<string, unknown>;

export interface QualityReport {
  score?: number | string;
  verdict?: string;
  features?: QualityRow[];
  videos?: QualityRow[];
  [key: string]: unknown;
}

export interface QualityResponse {
  command: string;
  report: QualityReport;
}

// ---------------------------------------------------------------------------
// Error handling.
// ---------------------------------------------------------------------------

/** Error thrown when an /api/* response is non-2xx or malformed. */
export class ApiError extends Error {
  readonly detail: string;
  readonly status: number;

  constructor(message: string, detail: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.detail = detail;
    this.status = status;
  }
}

interface ErrorBody {
  error?: string;
  detail?: string;
}

// ---------------------------------------------------------------------------
// Core request helpers.
// ---------------------------------------------------------------------------

async function request<T>(path: string, body: unknown, method = "POST"): Promise<T> {
  const init: RequestInit = {
    method,
    headers: {
      "Content-Type": "application/json",
      "X-Bagel-Token": TOKEN,
    },
  };
  if (body !== undefined) {
    init.body = JSON.stringify(body);
  }

  let res: Response;
  try {
    res = await fetch(path, init);
  } catch (e) {
    const detail = e instanceof Error ? e.message : String(e);
    throw new ApiError("Network error", detail, 0);
  }

  let data: unknown = null;
  const text = await res.text();
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      // Non-JSON body. Keep raw text for the error path below.
      if (!res.ok) {
        throw new ApiError(`HTTP ${res.status}`, text, res.status);
      }
    }
  }

  if (!res.ok) {
    const err = (data ?? {}) as ErrorBody;
    throw new ApiError(err.error ?? `HTTP ${res.status}`, err.detail ?? "", res.status);
  }

  return data as T;
}

function get<T>(path: string): Promise<T> {
  return request<T>(path, undefined, "GET");
}

// ---------------------------------------------------------------------------
// Typed endpoint functions.
// ---------------------------------------------------------------------------

export function getConfig(): Promise<ConfigResponse> {
  return get<ConfigResponse>("/api/config");
}

export function browse(rootId: string, subpath: string): Promise<BrowseResponse> {
  return request<BrowseResponse>("/api/browse", { root_id: rootId, subpath });
}

export function inspect(bags: string[]): Promise<InspectResponse> {
  return request<InspectResponse>("/api/inspect", { bags });
}

export interface ScaffoldOpts {
  robot_type?: string;
  task?: string;
  fps?: number;
}

export function scaffold(bags: string[], opts: ScaffoldOpts): Promise<ScaffoldResponse> {
  return request<ScaffoldResponse>("/api/scaffold", { bags, ...opts });
}

export function validateConfig(
  configYaml: string,
  bags: string[],
): Promise<ValidateConfigResponse> {
  return request<ValidateConfigResponse>("/api/validate-config", {
    config_yaml: configYaml,
    bags,
  });
}

export interface ConvertOpts {
  fps?: number;
  workers?: number;
  video_codec?: string;
}

export function convert(
  configYaml: string,
  bags: string[],
  output: string,
  opts: ConvertOpts,
): Promise<ConvertResponse> {
  return request<ConvertResponse>("/api/convert", {
    config_yaml: configYaml,
    bags,
    output,
    ...opts,
  });
}

export function convertStatus(jobId: string): Promise<ConvertStatus> {
  return get<ConvertStatus>(`/api/convert/${encodeURIComponent(jobId)}`);
}

export function validateDataset(dataset: string): Promise<ValidateDatasetResponse> {
  return request<ValidateDatasetResponse>("/api/validate-dataset", { dataset });
}

export function qualityReport(dataset: string): Promise<QualityResponse> {
  return request<QualityResponse>("/api/quality-report", { dataset });
}

/** Build the iframe URL for the HTML preview (token passed as a query param). */
export function previewUrl(dataset: string): string {
  const params = new URLSearchParams({ dataset });
  if (TOKEN) {
    params.set("token", TOKEN);
  }
  return `/api/preview?${params.toString()}`;
}
