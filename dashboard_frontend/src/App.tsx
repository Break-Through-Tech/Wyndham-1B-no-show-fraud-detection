
import { useEffect, useMemo, useState } from "react";

type FraudPrediction = {
  confirmation_number: string;
  member_number: string | null;
  fraud_risk_score: number;
  fraud_prediction: number;
  booking_timestamp: string | null;
  check_in_date: string | null;
  check_out_date: string | null;
  rate_code: string | null;
  booking_channel: string | null;
  qualification_code: string | null;
  points_earned: number | null;
  points_redeemed: number | null;
  room_revenue: number | null;
  member_enrollment_date: string | null;
};

type PredictionResponse = {
  dataset_id: string;
  total_reservations: number;
  flagged_reservations: number;
  flagged_rate: number;
  predictions: FraudPrediction[];
  daily_trend: { date: string; flagged_count: number }[];
};

type CaseStatus = "flagged" | "reviewing" | "cleared" | "escalated" | "confirmed_fraud";
type CaseRecord = { status: CaseStatus; notes: {text:string; created_at:string}[]; updated_at: string | null };
type QueueTab = "active" | "reviewing" | "reviewed";
const API = "http://127.0.0.1:8000";
const statusLabel: Record<CaseStatus,string> = {flagged:"Flagged",reviewing:"Under review",cleared:"Cleared · no further action",escalated:"Escalated",confirmed_fraud:"Confirmed fraud"};

type SortOrder = "high" | "low";

const CLOSED_STATUSES: CaseStatus[] = ["cleared", "escalated", "confirmed_fraud"];
const STORAGE_KEY = "wyndham.lastAnalysis.v1";

// Single source of truth for a row's status: queue tabs, counts, colors, export and modal all use this.
function statusOf(cases: Record<string, CaseRecord>, confirmation: string): CaseStatus {
  return cases[confirmation]?.status ?? "flagged";
}
function isClosed(status: CaseStatus) {
  return CLOSED_STATUSES.includes(status);
}
function tabForStatus(status: CaseStatus): QueueTab {
  return status === "flagged" ? "active" : status === "reviewing" ? "reviewing" : "reviewed";
}

function loadSavedAnalysis(): PredictionResponse | null {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const data = JSON.parse(raw);
    if (typeof data?.dataset_id === "string" && Array.isArray(data.predictions) && Array.isArray(data.daily_trend)
      && typeof data.total_reservations === "number" && typeof data.flagged_reservations === "number") return data;
  } catch { /* storage unavailable or corrupt: start empty */ }
  return null;
}
function saveAnalysis(data: PredictionResponse | null) {
  try {
    if (data) window.localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
    else window.localStorage.removeItem(STORAGE_KEY);
  } catch { /* ignore quota/private-mode errors */ }
}
async function fetchCases(datasetId: string): Promise<Record<string, CaseRecord>> {
  const response = await fetch(`${API}/cases/${encodeURIComponent(datasetId)}`);
  if (!response.ok) throw new Error("Could not load saved review decisions.");
  const saved = await response.json();
  return saved.cases ?? {};
}

function displayValue(value: string | number | null | undefined) {
  return value === null || value === undefined || value === "" ? "Not available" : String(value);
}

function formatDate(value: string | null) {
  return value ? value.slice(0, 10) : "Not available";
}

function formatNumber(value: number | null) {
  return value === null ? "Not available" : value.toLocaleString();
}

function Icon({
  name,
  size = 18,
}: {
  name:
    | "search"
    | "chevron"
    | "download"
    | "alert"
    | "users"
    | "check"
    | "shield";
  size?: number;
}) {
  const paths = {
    search: (
      <>
        <circle cx="10.5" cy="10.5" r="6.5" />
        <path d="m16 16 4 4" />
      </>
    ),
    chevron: <path d="m8 10 4 4 4-4" />,
    download: (
      <>
        <path d="M12 3v12m0 0 4-4m-4 4-4-4" />
        <path d="M4 20h16" />
      </>
    ),
    alert: (
      <>
        <path d="M12 3 2.8 20h18.4L12 3Z" />
        <path d="M12 9v4.5M12 17h.01" />
      </>
    ),
    users: (
      <>
        <circle cx="9" cy="8" r="3.5" />
        <path d="M2.5 20c.4-4 2.6-6 6.5-6s6.1 2 6.5 6M16 5.5a3 3 0 0 1 0 5.8M17 14c2.7.5 4.1 2.5 4.5 5" />
      </>
    ),
    check: <path d="m5 12 4 4L19 6" />,
    shield: (
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Zm-3-10 2 2 4-5" />
    ),
  };

  return (
    <svg
      aria-hidden="true"
      className="icon"
      width={size}
      height={size}
      viewBox="0 0 24 24"
    >
      {paths[name]}
    </svg>
  );
}

function KpiCard({
  icon,
  title,
  value,
  note,
  tone,
}: {
  icon: "alert" | "users";
  title: string;
  value: string;
  note: string;
  tone: "red" | "blue";
}) {
  return (
    <article className="kpi-card">
      <div className="kpi-title">
        <span className={`kpi-icon ${tone}`}>
          <Icon name={icon} size={16} />
        </span>
        {title}
      </div>
      <strong className="kpi-value">{value}</strong>
      <div className="receipt-divider" />
      <span className="kpi-note">{note}</span>
    </article>
  );
}

function RiskBadge({ score }: { score: number }) {
  const risk =
    score >= 0.8 ? "High" : score >= 0.5 ? "Medium" : "Low";

  return (
    <span className={`risk-badge risk-${risk.toLowerCase()}`}>
      <i />
      <span>{score.toFixed(2)}</span>
      <strong>{risk === "Medium" ? "MED" : risk.toUpperCase()}</strong>
    </span>
  );
}

function csvEscape(value: unknown) {
  let text = String(value ?? "");
  // Neutralize spreadsheet formulas in analyst-entered text (CSV injection).
  if (/^[=+@-]/.test(text) && typeof value === "string") text = `'${text}`;
  return `"${text.replace(/"/g, '""')}"`;
}

export default function App() {
  const [reservationsFile, setReservationsFile] =
    useState<File | null>(null);

  const [staysFile, setStaysFile] =
    useState<File | null>(null);

  const [predictionResults, setPredictionResults] =
    useState<PredictionResponse | null>(null);

  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");

  const [search, setSearch] = useState("");
  const [selectedDay, setSelectedDay] = useState<string | null>(null);
  const [sortOrder, setSortOrder] = useState<SortOrder>("high");
  const [selectedConfirmation, setSelectedConfirmation] =
    useState<string | null>(null);


  const [comment, setComment] = useState("");
  const [cases, setCases] = useState<Record<string,CaseRecord>>({});
  const [queueTab, setQueueTab] = useState<QueueTab>("active");
  const [caseBusy, setCaseBusy] = useState(false);
  const [caseError, setCaseError] = useState("");
  const [caseNotice, setCaseNotice] = useState("");

  const [restoredNotice, setRestoredNotice] = useState("");

  // After a browser refresh, restore the last analysis and re-read every decision/note from the backend.
  useEffect(() => {
    const saved = loadSavedAnalysis();
    if (!saved) return;
    let cancelled = false;
    fetchCases(saved.dataset_id)
      .then(savedCases => {
        if (cancelled) return;
        setCases(savedCases);
        setPredictionResults(saved);
        setRestoredNotice("Restored your last analysis. Decisions and notes were reloaded from the saved case store.");
      })
      .catch(() => {
        if (cancelled) return;
        // Do not show restored rows with default statuses: that would misrepresent saved decisions.
        setUploadError("Could not reach the backend to reload saved decisions. Start the API and refresh, or upload the files again.");
      });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!selectedConfirmation) return;
    const onKey = (event: KeyboardEvent) => { if (event.key === "Escape") void closeCase(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const flaggedRows = useMemo(() => {
    if (!predictionResults) return [];

    return predictionResults.predictions.filter(
      (row) => row.fraud_prediction === 1
    );
  }, [predictionResults]);

  const dailyTrend = predictionResults?.daily_trend ?? [];
  // Round up to a multiple of 3 so the four axis ticks are exact whole numbers.
  const chartMax = Math.ceil(Math.max(1, ...dailyTrend.map((day) => day.flagged_count)) / 3) * 3;

  const uniqueMembers = useMemo(() => {
    return new Set(
      flaggedRows
        .map((row) => row.member_number)
        .filter(
          (member): member is string =>
            member !== null && member !== ""
        )
    ).size;
  }, [flaggedRows]);

  const visibleRows = useMemo(() => {
    const query = search.trim().toLowerCase();

    return flaggedRows
      .filter(row => {
        return tabForStatus(statusOf(cases, row.confirmation_number)) === queueTab;
      })
      .filter(row => !selectedDay || row.check_in_date?.slice(0,10) === selectedDay)
      .filter((row) => {
        const content =
          `${row.confirmation_number} ${row.member_number ?? ""}`
            .toLowerCase();

        return !query || content.includes(query);
      })
      .sort((a, b) =>
        sortOrder === "high"
          ? b.fraud_risk_score - a.fraud_risk_score
          : a.fraud_risk_score - b.fraud_risk_score
      );
  }, [flaggedRows, search, sortOrder, queueTab, cases, selectedDay]);

  const selected = useMemo(() => {
    return flaggedRows.find(row => row.confirmation_number === selectedConfirmation) ?? null;
  }, [flaggedRows, selectedConfirmation]);

  async function handleUpload() {
    if (!reservationsFile || !staysFile) {
      setUploadError(
        "Please select both reservation and stay files."
      );
      return;
    }

    setIsUploading(true);
    setUploadError("");
    setPredictionResults(null);
    setRestoredNotice("");
    saveAnalysis(null);
    setSelectedConfirmation(null);
    setCases({});
    setQueueTab("active");
    setCaseError("");
    setComment("");
    setSearch("");
    setSelectedDay(null);

    try {
      const formData = new FormData();

      formData.append("reservations", reservationsFile);
      formData.append("stays", staysFile);

      const response = await fetch(
        `${API}/predict`,
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "Prediction request failed."
        );
      }

      if (
        typeof data.total_reservations !== "number" ||
        typeof data.flagged_reservations !== "number" ||
        !Array.isArray(data.predictions) ||
        !Array.isArray(data.daily_trend)
      ) {
        throw new Error(
          "The backend returned an unexpected response."
        );
      }

      if (typeof data.dataset_id !== "string") throw new Error("Backend needs the updated api.py (dataset_id missing).");
      const savedCases = await fetchCases(data.dataset_id);
      setCases(savedCases);
      setPredictionResults(data as PredictionResponse);
      saveAnalysis(data as PredictionResponse);
    } catch (error) {
      setUploadError(
        error instanceof Error
          ? error.message
          : "Something went wrong."
      );
    } finally {
      setIsUploading(false);
    }
  }

  function exportCsv() {
    if (!predictionResults) return;

    const header = [
      "Risk Score",
      "Confirmation Number",
      "Member Number",
      "Fraud Prediction",
      "Check-in Date", "Check-out Date", "Booking Date",
      "Rate Code", "Booking Channel", "Qualification Code",
      "Points Earned", "Points Redeemed", "Room Revenue",
      "Review Status", "Review Notes", "Member Enrollment Date",
    ];

    const rows = flaggedRows.map((row) => [
      row.fraud_risk_score,
      row.confirmation_number,
      row.member_number,
      row.fraud_prediction,
      row.check_in_date, row.check_out_date, row.booking_timestamp,
      row.rate_code, row.booking_channel, row.qualification_code,
      row.points_earned, row.points_redeemed, row.room_revenue,
      statusLabel[statusOf(cases, row.confirmation_number)],
      (cases[row.confirmation_number]?.notes ?? []).map(n => n.text).join(" | "),
      row.member_enrollment_date,
    ]);

    const csv = [header, ...rows]
      .map((line) => line.map(csvEscape).join(","))
      .join("\n");

    const url = URL.createObjectURL(
      new Blob([csv], {
        type: "text/csv;charset=utf-8",
      })
    );

    const link = document.createElement("a");
    link.href = url;
    link.download = "wyndham-fraud-review-cases.csv";
    link.click();

    URL.revokeObjectURL(url);
  }

  async function updateCase(confirmation: string, status?: CaseStatus, note?: string): Promise<boolean> {
    if (!predictionResults || caseBusy) return false;
    if (status && isClosed(status) && !note?.trim()) { setCaseError("Write a decision rationale in the note field before closing this case."); return false; }
    setCaseBusy(true);
    setCaseError("");
    setCaseNotice("");
    try {
      const response = await fetch(`${API}/cases/${predictionResults.dataset_id}/${encodeURIComponent(confirmation)}`, {
        method: "PATCH", headers: {"Content-Type":"application/json"},
        body: JSON.stringify({status: status ?? null, note: note?.trim() || null}),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Could not save review.");
      setCases(current => ({...current, [confirmation]: data.case}));
      if (note?.trim()) setComment("");
      setCaseNotice(`Saved ${new Date().toLocaleTimeString([], {hour: "numeric", minute: "2-digit"})}${status ? ` · moved to ${tabForStatus(status) === "active" ? "Active" : tabForStatus(status) === "reviewing" ? "Under Review" : "Reviewed"}` : ""}`);
      // Follow the case into the tab it now belongs to, so it never appears to vanish.
      if (status) setQueueTab(tabForStatus(status));
      return true;
    } catch (error) {
      setCaseError(error instanceof Error ? error.message : "Could not save review.");
      return false;
    } finally { setCaseBusy(false); }
  }

  async function closeCase() {
    if (caseBusy || !selected) return;
    // Save an unfinished note before closing. Keep the dialog open on failure.
    if (comment.trim()) {
      const saved = await updateCase(selected.confirmation_number, undefined, comment);
      if (!saved) return;
    }
    setSelectedConfirmation(null);
    setCaseNotice("");
    setCaseError("");
  }

  function openCase(confirmation: string) {
    setSelectedConfirmation(confirmation);
    setComment("");
    setCaseError("");
    setCaseNotice("");
  }

  const currentCase = selected ? cases[selected.confirmation_number] : undefined;
  const currentComments = currentCase?.notes ?? [];
  const currentStatus = currentCase?.status ?? "flagged";
  const tabCounts = { active: 0, reviewing: 0, reviewed: 0 };
  for (const r of flaggedRows) tabCounts[tabForStatus(statusOf(cases, r.confirmation_number))] += 1;
  const activeCount = tabCounts.active;
  const reviewingCount = tabCounts.reviewing;
  const reviewedCount = tabCounts.reviewed;
  const pointsExposure = flaggedRows.reduce((sum,row) => sum + Math.max(0, row.points_earned ?? 0),0);
  const completedCount = reviewedCount;
  const completedPct = flaggedRows.length ? 100 * completedCount / flaggedRows.length : 0;

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">
            W
          </div>

          <div className="brand-copy">
            <span>WYNDHAM</span>
            <strong>Fraud Risk Intelligence</strong>
          </div>
        </div>

        <div className="header-actions">
          <div className="model-status">
            <span aria-hidden="true">⚠️</span>
            Experimental Model · Non-Production
          </div>

          <button
            className="export-button"
            type="button"
            onClick={exportCsv}
            disabled={!predictionResults}
          >
            <Icon name="download" size={16} />
            Export CSV
          </button>
        </div>
      </header>

      <main className="dashboard">

        {/* MODEL 2 UPLOAD WORKSPACE */}

        <section
          className="model-workspace"
          aria-labelledby="upload-heading"
          style={{ marginBottom: "24px" }}
        >
          <div className="model-workspace-header">
            <div>
              <div className="model-title-line">
                <span className="kicker">
                  Fraud detection workspace
                </span>

              </div>

              <h2 id="upload-heading">
                Analyze Reservation Activity
              </h2>
            </div>
          </div>

          <div style={{ padding: "24px" }}>
            <p
              style={{
                marginTop: 0,
                color: "#64748b",
                lineHeight: 1.6,
              }}
            >
              Upload reservation and stay datasets to run
              fraud-risk predictions and generate an
              investigation queue.
            </p>

            <div
              style={{
                display: "grid",
                gridTemplateColumns:
                  "repeat(auto-fit, minmax(220px, 1fr))",
                gap: "20px",
                alignItems: "end",
                marginTop: "24px",
              }}
            >
              <label
                style={{
                  display: "grid",
                  gap: "10px",
                }}
              >
                <strong>Reservation Data</strong>

                <input
                  type="file"
                  accept=".csv,.xlsx"
                  onChange={(event) =>
                    setReservationsFile(
                      event.target.files?.[0] ?? null
                    )
                  }
                  style={{
                    width: "100%",
                    fontSize: "13px",
                  }}
                />

                <small style={{ color: "#64748b" }}>
                  CSV or Excel
                </small>
              </label>

              <label
                style={{
                  display: "grid",
                  gap: "10px",
                }}
              >
                <strong>Stay Data</strong>

                <input
                  type="file"
                  accept=".csv,.xlsx"
                  onChange={(event) =>
                    setStaysFile(
                      event.target.files?.[0] ?? null
                    )
                  }
                  style={{
                    width: "100%",
                    fontSize: "13px",
                  }}
                />

                <small style={{ color: "#64748b" }}>
                  CSV or Excel
                </small>
              </label>

              <button
                className="export-button"
                type="button"
                onClick={handleUpload}
                disabled={isUploading}
                style={{
                  justifyContent: "center",
                  background: "#3f70b5",
                  color: "#ffffff",
                  minHeight: "44px",
                  padding: "12px 20px",
                }}
              >
                {isUploading
                  ? "Analyzing Reservations..."
                  : "Run Fraud Detection"}
              </button>
            </div>

            {uploadError && (
              <p
                role="alert"
                style={{
                  color: "#b42318",
                  marginTop: "18px",
                }}
              >
                {uploadError}
              </p>
            )}

            {predictionResults && (
              <div
                role="status"
                style={{
                  marginTop: "24px",
                  paddingTop: "18px",
                  borderTop: "1px solid #d5dce6",
                }}
              >
                <strong>
                  <Icon name="check" size={16} />
                  {" "}Analysis Complete
                </strong>

                <p
                  style={{
                    color: "#475569",
                    marginBottom: 0,
                  }}
                >
                  {predictionResults.total_reservations.toLocaleString()}
                  {" "}reservations analyzed
                  {" · "}
                  {predictionResults.flagged_reservations.toLocaleString()}
                  {" "}flagged
                  {" · "}
                  {predictionResults.flagged_rate}%
                  {" "}flagged rate
                </p>
                {restoredNotice && <p style={{ color: "#475569", marginBottom: 0, fontSize: "13px" }}>{restoredNotice}</p>}
              </div>
            )}
          </div>
        </section>

        {/* DATASET SUMMARY */}

        <section
          className="date-section"
          aria-labelledby="review-date-heading"
        >
          <div className="section-heading-row">
            <div>
              <span className="kicker">
                Fraud analysis
              </span>

              <h1 id="review-date-heading">
                Reservation Fraud Risk Review
              </h1>
            </div>

            <span className="timezone">
              Uploaded dataset
            </span>
          </div>

          <p
            style={{
              color: "#64748b",
              marginTop: "8px",
            }}
          >
            {predictionResults
              ? "Showing results from the most recently uploaded files."
              : "Upload reservation and stay data to begin your review."}
          </p>
        </section>

        {/* OVERVIEW */}

        <section className="overview-grid">

          <article className="trend-card">
            <div className="card-heading">
              <div>
                <span className="kicker">Uploaded reservation data</span>
                <h2>Suspicious Reservation Volume</h2>
              </div>

            </div>
            <p style={{ fontSize: "12px", color: "#64748b", marginBottom: "16px" }}>
              {predictionResults
                ? "Flagged reservations by check-in date. Select a day to filter the queue; select it again to clear the filter."
                : "Upload reservation and stay files to display actual flagged counts by check-in date."}
            </p>
            <div className="chart">
              <div className="y-axis">
                <span>{chartMax}</span>
                <span>{chartMax * 2 / 3}</span>
                <span>{chartMax / 3}</span>
                <span>0</span>
              </div>
              <div className="chart-body">
                <div className="chart-grid"><i /><i /><i /><i /></div>
                <div className="bars">
                  {dailyTrend.length ? dailyTrend.map((day, index) => (
                    <button type="button" className={`bar-column day-button ${selectedDay === day.date ? "day-selected" : ""}`} key={day.date}
                      aria-label={`Filter cases for ${day.date}: ${day.flagged_count} flagged`}
                      title={`${day.date}: ${day.flagged_count} flagged — click to filter queue`}
                      onClick={() => {setSelectedDay(current => current === day.date ? null : day.date);}} >
                      <div className={`bar ${index === dailyTrend.length - 1 ? "active" : ""}`}
                        style={{ height: `${(day.flagged_count / chartMax) * 100}%` }} />
                    </button>
                  )) : <span style={{ alignSelf: "center", color: "#64748b", fontSize: "13px" }}>
                    No analysis yet
                  </span>}
                </div>
                <div className="x-axis">
                  {dailyTrend.length ? [0, 7, 14, 21, 29].map((index) => (
                    <span key={index}>{dailyTrend[index]?.date.slice(5)}</span>
                  )) : <span>Waiting for uploaded data</span>}
                </div>
              </div>
            </div>
          </article>

          <section
            className="summary-column"
            aria-label="Uploaded dataset summary"
          >
            <div className="summary-heading">
              <div>
                <span className="kicker">
                  Analysis summary
                </span>

                <h2>
                  Fraud Analysis Results
                </h2>
              </div>

              <span>
                Uploaded data
              </span>
            </div>

            <div className="kpi-grid">
              <KpiCard
                icon="alert"
                title="Flagged Reservations"
                value={
                  predictionResults
                    ? predictionResults.flagged_reservations.toLocaleString()
                    : "—"
                }
                note={
                  predictionResults
                    ? `${predictionResults.total_reservations.toLocaleString()} reservations analyzed`
                    : "Upload files to see results"
                }
                tone="red"
              />

              <KpiCard
                icon="users"
                title="Unique Suspicious Members"
                value={
                  predictionResults
                    ? uniqueMembers.toLocaleString()
                    : "—"
                }
                note={
                  predictionResults
                    ? "Distinct members among flagged reservations"
                    : "Upload files to see results"
                }
                tone="blue"
              />
            </div>
            <div className="points-overview">
              <div><span className="kicker">Potential loyalty points exposure</span><strong>{predictionResults ? pointsExposure.toLocaleString() : "—"}</strong><small>Points earned on flagged bookings; not verified losses or a cash valuation.</small></div>
              <div><span className="kicker">Investigation progress</span><strong>{completedPct.toFixed(1)}%</strong><div className="progress-track"><div style={{width:`${completedPct}%`}} /></div><small>{completedCount.toLocaleString()} of {flaggedRows.length.toLocaleString()} cases completed</small></div>
            </div>
          </section>
        </section>

        {/* INVESTIGATION FILTERS */}

        <section
          className="toolbar"
          aria-label="Queue search and filters"
        >
          <div className="queue-tabs" role="group" aria-label="Case status filters">
            <button type="button" className={queueTab === "active" ? "active" : ""} onClick={() => {setQueueTab("active");setSelectedConfirmation(null);}}>Active ({activeCount})</button>
            <button type="button" className={queueTab === "reviewing" ? "active" : ""} onClick={() => {setQueueTab("reviewing");setSelectedConfirmation(null);}}>Under Review ({reviewingCount})</button>
            <button type="button" className={queueTab === "reviewed" ? "active" : ""} onClick={() => {setQueueTab("reviewed");setSelectedConfirmation(null);}}>Reviewed ({reviewedCount})</button>
          </div>
          {selectedDay && <button className="day-clear" type="button" onClick={()=>setSelectedDay(null)}>Date: {selectedDay} ×</button>}
          <label className="queue-search">
            <Icon name="search" size={17} />

            <input
              aria-label="Search queue"
              placeholder="Search confirmation # or member ID..."
              value={search}
              onChange={(event) =>
                setSearch(event.target.value)
              }
            />
          </label>

          <label className="filter-control">
            <span>
              {sortOrder === "high"
                ? "Risk Score: High to Low"
                : "Risk Score: Low to High"}
            </span>

            <Icon name="chevron" size={14} />

            <select
              aria-label="Sort reservations"
              value={sortOrder}
              onChange={(event) =>
                setSortOrder(
                  event.target.value as SortOrder
                )
              }
            >
              <option value="high">
                Risk Score: High to Low
              </option>

              <option value="low">
                Risk Score: Low to High
              </option>
            </select>
          </label>
        </section>

        {/* INVESTIGATION WORKSPACE */}

        <div className="workspace">

          <section
            className="queue-card"
            aria-labelledby="queue-heading"
          >
            <div className="queue-heading">
              <div>
                <span className="kicker">
                  Prioritized queue
                </span>

                <h2 id="queue-heading">
                  {queueTab === "reviewed" ? "Reviewed Cases" : queueTab === "reviewing" ? "Cases Under Review" : "Detailed Investigation Queue"}
                </h2>
              </div>

              <span>
                <strong>{visibleRows.length}</strong>
                {" "}records shown
              </span>
            </div>

            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Risk / Decision</th>
                    <th>Confirmation #</th>
                    <th>Member ID</th>
                    <th>Review Status</th>
                    <th>Action</th>
                  </tr>
                </thead>

                <tbody>
                  {visibleRows.map((row) => (
                    <tr
                      key={row.confirmation_number}
                      className={
                        selected?.confirmation_number ===
                        row.confirmation_number
                          ? "selected-row"
                          : ""
                      }
                      onClick={() => { openCase(row.confirmation_number); }}
                    >
                      <td>
                        {statusOf(cases, row.confirmation_number) === "flagged"
                          ? <RiskBadge score={row.fraud_risk_score} />
                          : <span className={`decision-pill ${statusOf(cases, row.confirmation_number)}`}>{statusOf(cases, row.confirmation_number) === "cleared" ? "✓ " : ""}{statusLabel[statusOf(cases, row.confirmation_number)].split(" · ")[0]}</span>}
                      </td>

                      <td>
                        <button
                          className="mono-link"
                          type="button"
                          onClick={() => { openCase(row.confirmation_number); }}
                        >
                          {row.confirmation_number}
                        </button>
                      </td>

                      <td>
                        <span className="mono-link member-link">
                          {row.member_number ?? "Not available"}
                        </span>
                      </td>

                      <td>
                        <strong className={`explanation status-${statusOf(cases, row.confirmation_number)}`}>
                          {statusLabel[statusOf(cases, row.confirmation_number)]}
                        </strong>

                        <span className="cell-detail">Original model risk: {(row.fraud_risk_score * 100).toFixed(0)}%{(cases[row.confirmation_number]?.notes.length ?? 0) > 0 ? ` · ${cases[row.confirmation_number].notes.length} note${cases[row.confirmation_number].notes.length === 1 ? "" : "s"}` : ""}</span>
                      </td>

                      <td>
                        <button
                          className="review-button"
                          type="button"
                          onClick={(event) => {
                            event.stopPropagation();
                            openCase(row.confirmation_number);
                          }}
                        >
                          View case
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>

              {!predictionResults && (
                <div className="empty-state">
                  Upload reservation and stay files to
                  generate the investigation queue.
                </div>
              )}

              {predictionResults &&
                flaggedRows.length === 0 && (
                  <div className="empty-state">
                    No reservations were flagged in
                    this uploaded dataset.
                  </div>
                )}

              {predictionResults &&
                flaggedRows.length > 0 &&
                visibleRows.length === 0 && (
                  <div className="empty-state">
                    No cases match this view or search.
                  </div>
                )}
            </div>

            <div className="table-footer">
              <span>
                Showing {visibleRows.length} of{" "}
                {flaggedRows.length} total flagged reservations
              </span>

              <span>
                {predictionResults
                  ? "Latest uploaded analysis"
                  : "Awaiting upload"}
              </span>
            </div>

            <p className="compliance-note">Experimental screening on uploaded data. Analyst decisions are saved locally; no operational action is performed.</p>
          </section>

          {/* Focused analyst workspace: full-size case view, not a narrow sidebar. */}
          {selectedConfirmation && selected && (
            <div className="case-modal-backdrop" onMouseDown={(event) => {if(event.target === event.currentTarget) void closeCase();}}>
              <section className="case-modal" role="dialog" aria-modal="true" aria-labelledby="case-title">
                <header className="case-modal-header">
                  <div>
                    <span className="kicker">Reservation investigation</span>
                    <h2 id="case-title">{selected.confirmation_number}</h2>
                    <p>Member {displayValue(selected.member_number)} · {statusLabel[currentStatus]}</p>
                  </div>
                  <button type="button" className="case-close" onClick={() => void closeCase()} disabled={caseBusy} aria-label="Save and close case details">✕</button>
                </header>
                <div className="case-modal-body">
                  <div className="case-review-main">
                    <div className="case-section">
                      <h3>Reservation & stay details</h3>
                      <p className="case-help">Information from the uploaded reservation and stay files. Missing values are shown as unavailable.</p>
                      <div className="case-facts-grid">
                        <div><span>Booking date</span><strong>{formatDate(selected.booking_timestamp)}</strong></div>
                        <div><span>Check-in</span><strong>{formatDate(selected.check_in_date)}</strong></div>
                        <div><span>Check-out</span><strong>{formatDate(selected.check_out_date)}</strong></div>
                        <div><span>Booking channel</span><strong>{displayValue(selected.booking_channel)}</strong></div>
                        <div><span>Rate code</span><strong>{displayValue(selected.rate_code)}</strong></div>
                        <div><span>Stay qualification</span><strong>{displayValue(selected.qualification_code)}</strong></div>
                        <div><span>Points earned</span><strong>{formatNumber(selected.points_earned)}</strong></div>
                        <div><span>Points redeemed</span><strong>{formatNumber(selected.points_redeemed)}</strong></div>
                        <div><span>Room revenue</span><strong>{formatNumber(selected.room_revenue)}</strong></div>
                      </div>
                    </div>
                    <div className="case-section">
                      <h3>Evidence for analyst review</h3>
                      <p className="case-help">These are review prompts based on available fields, not verified reasons for the prediction.</p>
                      <div className="case-checks">
                        <div><strong>Booking lead time</strong><span>{selected.booking_timestamp && selected.check_in_date ? `${Math.round((new Date(selected.check_in_date).getTime() - new Date(selected.booking_timestamp).getTime()) / 86400000)} days before check-in` : "Not available"}</span></div>
                        <div><strong>Length of stay</strong><span>{selected.check_in_date && selected.check_out_date ? `${Math.round((new Date(selected.check_out_date).getTime() - new Date(selected.check_in_date).getTime()) / 86400000)} night(s)` : "Not available"}</span></div>
                        <div><strong>Loyalty activity</strong><span>{formatNumber(selected.points_earned)} points earned · {formatNumber(selected.points_redeemed)} redeemed</span></div>
                        <div><strong>Qualification & rate</strong><span>{displayValue(selected.qualification_code)} · {displayValue(selected.rate_code)}</span></div>
                        <div><strong>Account tenure at booking</strong><span>{selected.member_enrollment_date && selected.booking_timestamp ? `${Math.round((new Date(selected.booking_timestamp).getTime() - new Date(selected.member_enrollment_date).getTime()) / 86400000)} days` : "Not available"}</span></div>
                        <div><strong>Potential points exposure</strong><span>{formatNumber(selected.points_earned)} earned points associated with this flagged booking (not confirmed loss)</span></div>
                      </div>
                      <p className="case-help">Confirm suspicious activity against policy and source records before concluding fraud. This prototype does not provide member history, verified fraud labels, or individual feature attributions.</p>
                    </div>
                  </div>
                  <aside className="case-review-actions">
                    <div className="case-section">
                      <span className="panel-label">Fraud analysis</span>
                      <div className="case-score"><span className={`decision-pill ${currentStatus}`}>{statusLabel[currentStatus]}</span><span className="model-risk">Model risk: {(selected.fraud_risk_score * 100).toFixed(0)}%</span></div>
                      <strong>{currentStatus === "cleared" ? "Cleared — no further action" : currentStatus === "confirmed_fraud" ? "Fraud confirmed by analyst" : currentStatus === "escalated" ? "Escalated for further investigation" : currentStatus === "reviewing" ? "Investigation in progress" : "Flagged for analyst review"}</strong>
                      <p className="case-help">Risk score is a screening output, not a confirmed fraud finding. It does not change when an analyst updates the case.</p>
                    </div>
                    <div className="case-section">
                      <label className="case-status-label" htmlFor="case-status">Review decision</label>
                      <select id="case-status" className="case-status-select" value={currentStatus} disabled={caseBusy} onChange={e => void updateCase(selected.confirmation_number, e.target.value as CaseStatus, comment)}>
                        <option value="flagged">Not started</option>
                        <option value="reviewing">Under review</option>
                        <option value="cleared">Cleared / no further action</option>
                        <option value="escalated">Escalated / further investigation</option>
                        <option value="confirmed_fraud">Confirmed fraud (documented evidence)</option>
                      </select>
                      <p className="case-help">Select Under review before a final decision. For Clear, Escalate, or Confirm fraud, enter a rationale below. Confirmed fraud is a recorded analyst decision only, not a Wyndham system update.</p>
                      {caseError && <p className="case-error" role="alert">{caseError}</p>}
                      {caseNotice && <p className="case-saved" role="status">✓ {caseNotice} · Decision and notes retained after closing</p>}
                    </div>
                    <div className="case-section">
                      <h3>Analyst notes</h3>
                      <div className="case-notes">{currentComments.length ? currentComments.map((entry,i) => <div key={i} className="case-note"><small>{new Date(entry.created_at).toLocaleString()}</small><p>{entry.text}</p></div>) : <p className="case-help">No notes yet.</p>}</div>
                      <textarea aria-label="Analyst note" value={comment} onChange={e=>{setComment(e.target.value);setCaseNotice("");}} rows={4} placeholder="Record evidence, follow-up, or decision rationale..." />
                      <button type="button" className="primary-action case-save" disabled={caseBusy || !comment.trim()} onClick={()=>void updateCase(selected.confirmation_number,undefined,comment)}>Save note</button>
                    </div>
                  </aside>
                </div>
                <footer className="case-modal-footer">Local demonstration only · No accounts, loyalty points, or bookings are changed. <button type="button" disabled={caseBusy} onClick={()=>void closeCase()}>{caseBusy ? "Saving…" : comment.trim() ? "Save note & close" : "Close"}</button></footer>
              </section>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}


