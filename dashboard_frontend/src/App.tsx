
import { useMemo, useState } from "react";

type FraudPrediction = {
  confirmation_number: string;
  member_number: string | null;
  fraud_risk_score: number;
  fraud_prediction: number;
};

type PredictionResponse = {
  total_reservations: number;
  flagged_reservations: number;
  flagged_rate: number;
  predictions: FraudPrediction[];
};

type ModelView = "details" | "comparison";
type SortOrder = "high" | "low";

const trendData = [
  26, 18, 31, 23, 37, 19, 28, 41, 24, 34,
  21, 39, 29, 44, 32, 25, 38, 46, 35, 22,
  39, 43, 31, 48, 36, 42, 57, 39, 51, 33,
];

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
  return `"${String(value ?? "").replaceAll('"', '""')}"`;
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
  const [sortOrder, setSortOrder] = useState<SortOrder>("high");
  const [selectedConfirmation, setSelectedConfirmation] =
    useState<string | null>(null);

  const [modelView, setModelView] =
    useState<ModelView>("comparison");

  const [comment, setComment] = useState("");
  const [savedComments, setSavedComments] =
    useState<Record<string, string[]>>({});

  const [decisions, setDecisions] =
    useState<Record<string, string>>({});

  const flaggedRows = useMemo(() => {
    if (!predictionResults) return [];

    return predictionResults.predictions.filter(
      (row) => row.fraud_prediction === 1
    );
  }, [predictionResults]);

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
  }, [flaggedRows, search, sortOrder]);

  const selected = useMemo(() => {
    return (
      flaggedRows.find(
        (row) =>
          row.confirmation_number === selectedConfirmation
      ) ??
      flaggedRows[0] ??
      null
    );
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
    setSelectedConfirmation(null);
    setSearch("");

    try {
      const formData = new FormData();

      formData.append("reservations", reservationsFile);
      formData.append("stays", staysFile);

      const response = await fetch(
        "http://127.0.0.1:8000/predict",
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
        !Array.isArray(data.predictions)
      ) {
        throw new Error(
          "The backend returned an unexpected response."
        );
      }

      setPredictionResults(data as PredictionResponse);
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
    ];

    const rows = visibleRows.map((row) => [
      row.fraud_risk_score,
      row.confirmation_number,
      row.member_number,
      row.fraud_prediction,
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
    link.download = "wyndham-model2-flagged-reservations.csv";
    link.click();

    URL.revokeObjectURL(url);
  }

  function saveComment() {
    if (!selected || !comment.trim()) return;

    const key = selected.confirmation_number;

    setSavedComments((current) => ({
      ...current,
      [key]: [
        ...(current[key] ?? []),
        comment.trim(),
      ],
    }));

    setComment("");
  }

  function saveDecision(value: string) {
    if (!selected) return;

    setDecisions((current) => ({
      ...current,
      [selected.confirmation_number]: value,
    }));
  }

  const currentComments = selected
    ? savedComments[selected.confirmation_number] ?? []
    : [];

  const currentDecision = selected
    ? decisions[selected.confirmation_number] ?? ""
    : "";

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
                <span className="experimental-badge">
                  Model 2 · Experimental
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
                Model 2 analysis
              </span>

              <h1 id="review-date-heading">
                Reservation Fraud Risk Review
              </h1>
            </div>

            <span className="timezone">
              Uploaded dataset · Experimental
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
                <span className="kicker">
                  Illustrative activity trend
                </span>

                <h2>
                  Suspicious Reservation Volume
                </h2>
              </div>

              <span className="experimental-badge">
                Demo visualization
              </span>
            </div>

            <p
              style={{
                fontSize: "12px",
                color: "#64748b",
                marginBottom: "16px",
              }}
            >
              Example chart only. Historical daily
              counts are not provided by the prediction API.
            </p>

            <div className="chart">
              <div className="y-axis">
                <span>60</span>
                <span>40</span>
                <span>20</span>
                <span>0</span>
              </div>

              <div className="chart-body">
                <div className="chart-grid">
                  <i />
                  <i />
                  <i />
                  <i />
                </div>

                <div className="bars">
                  {trendData.map((value, index) => (
                    <div
                      className="bar-column"
                      key={index}
                    >
                      <div
                        className={`bar ${
                          index === 28 ? "active" : ""
                        }`}
                        style={{
                          height: `${(value / 60) * 100}%`,
                        }}
                      />
                    </div>
                  ))}
                </div>

                <div className="x-axis">
                  <span>Day 1</span>
                  <span>Day 8</span>
                  <span>Day 15</span>
                  <span>Day 22</span>
                  <strong>Day 30</strong>
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
                  Model 2 Results
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
          </section>
        </section>

        {/* MODEL COMPARISON */}

        <section
          className="model-workspace"
          aria-labelledby="model-workspace-heading"
        >
          <div className="model-workspace-header">
            <div>
              <div className="model-title-line">
                <span className="kicker">
                  Model comparison &amp; details
                </span>

                <span className="experimental-badge">
                  Experimental
                </span>
              </div>

              <h2 id="model-workspace-heading">
                Model Architecture &amp; Benchmark Comparison
              </h2>
            </div>

            <div
              className="model-tabs"
              role="tablist"
              aria-label="Model intelligence views"
            >
              <button
                aria-selected={modelView === "details"}
                className={
                  modelView === "details" ? "active" : ""
                }
                onClick={() => setModelView("details")}
                role="tab"
                type="button"
              >
                Model Details · Top Signals
              </button>

              <button
                aria-selected={modelView === "comparison"}
                className={
                  modelView === "comparison" ? "active" : ""
                }
                onClick={() => setModelView("comparison")}
                role="tab"
                type="button"
              >
                Benchmark Comparison
              </button>
            </div>
          </div>

          <div className="model-notice">
            <Icon name="alert" size={15} />

            <span>
              <strong>
                Experimental baseline on synthetic May 2026 data.
              </strong>
              {" "}Reported model performance has not been
              validated on production Wyndham data.
              Operational feature availability requires verification.
            </span>
          </div>

          {modelView === "details" ? (
            <div
              className="signal-tape-panel"
              role="tabpanel"
            >
              <div className="signal-tape-heading">
                <span className="model-field-label">
                  Observed signals · Synthetic May test data
                </span>

                <span>
                  Historical analysis, not individual explanations
                </span>
              </div>

              <div className="signal-tape">
                {[
                  [
                    "1",
                    "Qualification Code QXY",
                    "100% Fraud Rate",
                    "320/320 events in the analyzed synthetic subset",
                  ],
                  [
                    "2",
                    "Points Earned >20,000",
                    "8.36% Fraud Rate",
                    "Compared with the lower-point cohort",
                  ],
                  [
                    "3",
                    "Account Tenure 8–28 Days",
                    "7.44% Fraud Rate",
                    "Elevated concentration in synthetic data",
                  ],
                  [
                    "4",
                    "Rate Code SDC4",
                    "0.33% Fraud Rate",
                    "876 of 1,600 May fraud cases",
                  ],
                  [
                    "5",
                    "Same-Day Booking",
                    "1.57% Fraud Rate",
                    "Compared with longer lead times",
                  ],
                  [
                    "6",
                    "One-Night Stays",
                    "90% of Fraud Events",
                    "Concentration observed in test data",
                  ],
                  [
                    "7",
                    "Web & App Channels",
                    "100% of Fraud",
                    "Observed in synthetic May cases",
                  ],
                ].map(
                  ([rank, label, metric, context]) => (
                    <article
                      className="signal-chip"
                      key={rank}
                    >
                      <b>{rank}</b>
                      <span>{label}</span>
                      <strong>{metric}</strong>
                      <small>{context}</small>
                    </article>
                  )
                )}
              </div>
            </div>
          ) : (
            <div
              className="model-comparison-grid"
              role="tabpanel"
            >
              <article className="architecture-card">
                <div className="architecture-heading">
                  <span>Model A · Experimental</span>

                  <h3>
                    Model 1
                    <small> (Booking-Time Risk)</small>
                  </h3>
                </div>

                <dl>
                  <div>
                    <dt>Prediction stage</dt>
                    <dd>
                      At Booking Creation
                      <span>Pre-Stay Risk Detection</span>
                    </dd>
                  </div>

                  <div>
                    <dt>Feature count</dt>
                    <dd>
                      <strong>13 Features</strong>
                      <span>
                        Booking behavior, account tenure,
                        and historical reservation activity
                      </span>
                    </dd>
                  </div>

                  <div>
                    <dt>Performance</dt>
                    <dd>
                      <strong>
                        49.0% Precision · 100.0% Recall
                      </strong>
                      <span>
                        326 false positives in held-out synthetic data
                      </span>
                    </dd>
                  </div>

                  <div>
                    <dt>Operational role</dt>
                    <dd>
                      Early warning for suspicious bookings
                    </dd>
                  </div>
                </dl>
              </article>

              <article className="architecture-card">
                <div className="architecture-heading">
                  <span>Model B · Experimental</span>

                  <h3>
                    Conservative Model 1
                    <small> (Booking-Time Only)</small>
                  </h3>
                </div>

                <dl>
                  <div>
                    <dt>Prediction stage</dt>
                    <dd>
                      At Booking Creation
                      <span>No Post-Stay Features</span>
                    </dd>
                  </div>

                  <div>
                    <dt>Feature count</dt>
                    <dd>
                      <strong>9 Features</strong>
                      <span>
                        Booking dynamics and account age
                      </span>
                    </dd>
                  </div>

                  <div>
                    <dt>Performance</dt>
                    <dd>
                      <strong>
                        46.7% Precision · 100.0% Recall
                      </strong>
                      <span>
                        357 false positives in synthetic evaluation
                      </span>
                    </dd>
                  </div>

                  <div>
                    <dt>Operational role</dt>
                    <dd>
                      Conservative booking-time risk screening
                    </dd>
                  </div>
                </dl>
              </article>

              <article className="architecture-card recommended">
                <div className="recommendation-label">
                  Recommended benchmark
                </div>

                <div className="architecture-heading">
                  <span>Model C · Experimental</span>

                  <h3>
                    Model 2
                    <small> (Pre-Redemption Intervention)</small>
                  </h3>
                </div>

                <dl>
                  <div>
                    <dt>Prediction stage</dt>
                    <dd>
                      Post-Stay / Outcome Posted
                      <span>
                        Before potential loyalty-point redemption
                      </span>
                    </dd>
                  </div>

                  <div>
                    <dt>Feature count</dt>
                    <dd>
                      <strong>18 Features</strong>
                      <span>
                        Includes current qualification outcomes,
                        points earned, and room revenue
                      </span>
                    </dd>
                  </div>

                  <div>
                    <dt>Performance</dt>
                    <dd>
                      <strong>
                        97.5% Precision · 100.0% Recall
                      </strong>
                      <span>
                        8 false positives in held-out synthetic data
                      </span>
                    </dd>
                  </div>

                  <div>
                    <dt>Operational role</dt>
                    <dd>
                      Identify potentially suspicious loyalty activity
                      for analyst review
                    </dd>
                  </div>
                </dl>
              </article>
            </div>
          )}
        </section>

        {/* INVESTIGATION FILTERS */}

        <section
          className="toolbar"
          aria-label="Queue search and filters"
        >
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
                  Detailed Investigation Queue
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
                    <th>Risk Score</th>
                    <th>Confirmation #</th>
                    <th>Member ID</th>
                    <th>Model Prediction</th>
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
                      onClick={() =>
                        setSelectedConfirmation(
                          row.confirmation_number
                        )
                      }
                    >
                      <td>
                        <RiskBadge
                          score={row.fraud_risk_score}
                        />
                      </td>

                      <td>
                        <button
                          className="mono-link"
                          type="button"
                          onClick={() =>
                            setSelectedConfirmation(
                              row.confirmation_number
                            )
                          }
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
                        <strong className="explanation">
                          Flagged for review
                        </strong>

                        <span className="cell-detail">
                          Model 2 prediction
                        </span>
                      </td>

                      <td>
                        <button
                          className="review-button"
                          type="button"
                          onClick={(event) => {
                            event.stopPropagation();
                            setSelectedConfirmation(
                              row.confirmation_number
                            );
                          }}
                        >
                          Review
                          <Icon name="chevron" size={12} />
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
                    No reservations match your search.
                  </div>
                )}
            </div>

            <div className="table-footer">
              <span>
                Showing {visibleRows.length} of{" "}
                {flaggedRows.length} flagged reservations
              </span>

              <span>
                {predictionResults
                  ? "Latest uploaded analysis"
                  : "Awaiting upload"}
              </span>
            </div>

            <p className="compliance-note">
              <strong>Experimental Review:</strong>
              {" "}Predictions are generated by Model 2.
              Analyst notes and decisions in this prototype
              are local demonstration state only.
              No operational action is performed.
            </p>
          </section>

          {/* INVESTIGATION DETAILS */}

          <aside
            className="investigation-panel"
            aria-labelledby="investigation-heading"
          >
            <div className="panel-top">
              <span className="kicker">
                Investigation
              </span>

              <h2 id="investigation-heading">
                {selected
                  ? `Reservation ${selected.confirmation_number}`
                  : "Reservation Details"}
              </h2>

              <p className="member-overview">
                {selected
                  ? `Member ${selected.member_number ?? "Not available"}`
                  : "Select a flagged reservation to review."}
              </p>
            </div>

            {selected ? (
              <>
                <div className="timeline-section">
                  <div className="panel-label">
                    Model 2 Assessment
                  </div>

                  <div
                    style={{
                      padding: "16px 0",
                    }}
                  >
                    <p>
                      <strong>
                        Fraud Risk Score
                      </strong>
                    </p>

                    <div style={{ marginBottom: "16px" }}>
                      <RiskBadge
                        score={selected.fraud_risk_score}
                      />
                    </div>

                    <p>
                      <strong>
                        Prediction:
                      </strong>
                      {" "}Flagged for analyst review
                    </p>

                    <p
                      style={{
                        fontSize: "13px",
                        color: "#64748b",
                        lineHeight: 1.6,
                      }}
                    >
                      This score comes from the saved
                      Model 2 pipeline. A positive
                      prediction is a screening result,
                      not proof of fraud.
                    </p>
                  </div>
                </div>

                <div className="signal-section">
                  <div className="panel-label">
                    Model Explanation
                  </div>

                  <p
                    style={{
                      fontSize: "13px",
                      color: "#64748b",
                      lineHeight: 1.6,
                    }}
                  >
                    Individual feature contributions
                    and SHAP explanations are not
                    currently returned by the backend.
                    This section can be connected
                    when explanation data is available.
                  </p>
                </div>

                <div className="decision-section">
                  <div className="panel-label">
                    Analyst Review · Prototype
                  </div>

                  {currentDecision && (
                    <div
                      className="decision-confirmation"
                      role="status"
                    >
                      <Icon name="check" size={14} />
                      {" "}Local review status: {currentDecision}
                    </div>
                  )}

                  <div className="decision-actions">
                    <button
                      className="primary-action"
                      type="button"
                      onClick={() =>
                        saveDecision("Escalated for review")
                      }
                    >
                      <Icon name="shield" size={15} />
                      Escalate for Review
                    </button>

                    <button
                      className="secondary-action"
                      type="button"
                      onClick={() =>
                        saveDecision("Marked false positive")
                      }
                    >
                      Mark False Positive
                    </button>
                  </div>

                  <p
                    style={{
                      fontSize: "12px",
                      color: "#64748b",
                      lineHeight: 1.6,
                    }}
                  >
                    Demo actions only. No points
                    are frozen, accounts suspended,
                    or backend records updated.
                  </p>

                  <div className="audit-record">
                    <div className="audit-heading">
                      <span className="panel-label">
                        Review Comments
                      </span>

                      <span>
                        {currentComments.length} local notes
                      </span>
                    </div>

                    {currentComments.map(
                      (entry, index) => (
                        <div
                          className="audit-entry"
                          key={index}
                        >
                          <div>
                            <strong>
                              Prototype Analyst
                            </strong>
                            <time>Current session</time>
                          </div>

                          <p>{entry}</p>
                        </div>
                      )
                    )}

                    <label className="comment-box">
                      <span>
                        Add analyst comment
                      </span>

                      <textarea
                        aria-label="Add analyst comment"
                        value={comment}
                        onChange={(event) =>
                          setComment(event.target.value)
                        }
                        placeholder="Add context for this review..."
                        rows={3}
                      />
                    </label>

                    <button
                      className="save-comment"
                      type="button"
                      disabled={!comment.trim()}
                      onClick={saveComment}
                    >
                      Save Comment
                    </button>

                    <p className="feedback-disclosure">
                      Comments and decisions are
                      stored only in the current
                      browser session's React state.
                      They are not a persistent audit
                      trail and do not retrain the model.
                    </p>
                  </div>
                </div>
              </>
            ) : (
              <div
                className="empty-state"
                style={{
                  padding: "32px 20px",
                }}
              >
                {predictionResults
                  ? "No flagged reservation is available for investigation."
                  : "Run Model 2 to populate this investigation panel."}
              </div>
            )}
          </aside>
        </div>
      </main>
    </div>
  );
}


