CREATE TABLE IF NOT EXISTS predictions (
  request_id UUID PRIMARY KEY, ts TIMESTAMPTZ NOT NULL DEFAULT now(),
  model_version INT NOT NULL, slot TEXT, features JSONB NOT NULL,
  prediction SMALLINT NOT NULL, probability DOUBLE PRECISION NOT NULL, latency_ms DOUBLE PRECISION);
CREATE TABLE IF NOT EXISTS labels (
  request_id UUID PRIMARY KEY, label SMALLINT NOT NULL, ts TIMESTAMPTZ NOT NULL DEFAULT now(),
  exported_at TIMESTAMPTZ, exported_event_id BIGINT);
CREATE TABLE IF NOT EXISTS data_batches (
  batch_id TEXT PRIMARY KEY, sha256 TEXT NOT NULL, status TEXT NOT NULL, ts TIMESTAMPTZ NOT NULL DEFAULT now(), report JSONB);
CREATE TABLE IF NOT EXISTS monitoring_results (
  id BIGSERIAL PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL, model_version INT,
  window_end TIMESTAMPTZ, n_rows INT, n_labelled INT, value DOUBLE PRECISION, details JSONB, ts TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS retrain_events (
  id BIGSERIAL PRIMARY KEY, trigger TEXT NOT NULL, status TEXT NOT NULL, data_version_id TEXT, candidate_version INT,
  started_at TIMESTAMPTZ NOT NULL DEFAULT now(), finished_at TIMESTAMPTZ, reason TEXT, gate_report JSONB);
CREATE UNIQUE INDEX IF NOT EXISTS retrain_one_running ON retrain_events ((true)) WHERE status = 'RUNNING';
CREATE TABLE IF NOT EXISTS deployments (
  id BIGSERIAL PRIMARY KEY, model_version INT NOT NULL, status TEXT NOT NULL, slot TEXT, image_tag TEXT,
  started_at TIMESTAMPTZ NOT NULL DEFAULT now(), finished_at TIMESTAMPTZ, details JSONB);
CREATE TABLE IF NOT EXISTS events (
  id BIGSERIAL PRIMARY KEY, ts TIMESTAMPTZ NOT NULL DEFAULT now(), severity TEXT NOT NULL, source TEXT NOT NULL,
  message TEXT NOT NULL, details JSONB);
CREATE TABLE IF NOT EXISTS gateway_state (
  id BOOLEAN PRIMARY KEY DEFAULT true CHECK (id), active TEXT NOT NULL DEFAULT 'blue', updated_at TIMESTAMPTZ NOT NULL DEFAULT now());
INSERT INTO gateway_state (id, active) VALUES (true, 'blue') ON CONFLICT DO NOTHING;
