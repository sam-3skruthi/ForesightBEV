import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { get } from "./api";
import { useApi } from "./hooks";
import type { FrameRecord, RunDetail, Sequence, TimelineRow } from "./types";

interface Selection {
  page: string; run: string | null;
  /** The frame shown: the last one loaded. `target` is the one asked for. */
  frame: number; target: number;
  sequences: Sequence[] | null; sequencesError: string | null;
  detail: RunDetail | null; timeline: TimelineRow[] | null; record: FrameRecord | null; recordError: string | null;
  sequence: Sequence | null; playing: boolean;
  setPlaying: (p: boolean) => void; go: (page: string) => void; setRun: (run: string) => void; setFrame: (frame: number) => void; reload: () => void;
}

const Context = createContext<Selection | null>(null);
export const useSelection = () => { const c = useContext(Context); if (!c) throw new Error("no selection context"); return c; };

function parseHash() {
  const [path, query] = window.location.hash.replace(/^#\/?/, "").split("?");
  const params = new URLSearchParams(query ?? "");
  return { page: path || "overview", run: params.get("run"), frame: Number(params.get("frame") ?? 0) || 0 };
}

export function SelectionProvider({ children }: { children: ReactNode }) {
  const [route, setRoute] = useState(parseHash);
  const [playing, setPlaying] = useState(false);
  const [refresh, setRefresh] = useState(0);
  const sequences = useApi<Sequence[]>("/sequences", refresh);
  const run = route.run ?? sequences.data?.flatMap((s) => s.runs).find((r) => r.id === "showcase_scene-0916")?.id ?? sequences.data?.flatMap((s) => s.runs)[0]?.id ?? null;
  const detail = useApi<RunDetail>(run ? `/runs/${run}` : null, refresh);
  const timeline = useApi<TimelineRow[]>(run ? `/runs/${run}/timeline` : null, refresh);
  const [record, setRecord] = useState<{ run: string | null; value: FrameRecord | null; error: string | null }>({ run: null, value: null, error: null });

  useEffect(() => { const on = () => setRoute(parseHash()); window.addEventListener("hashchange", on); return () => window.removeEventListener("hashchange", on); }, []);

  const write = useCallback((page: string, nextRun: string | null, frame: number) => {
    const query = new URLSearchParams();
    if (nextRun) query.set("run", nextRun);
    query.set("frame", String(frame));
    window.history.replaceState(null, "", `#/${page}?${query}`);
    setRoute({ page, run: nextRun, frame });
  }, []);

  const target = timeline.data ? Math.min(route.frame, Math.max(timeline.data.length - 1, 0)) : route.frame;
  useEffect(() => {
    if (!run) return;
    const controller = new AbortController();
    get<FrameRecord>(`/runs/${run}/frames/${target}/record`, controller.signal)
      .then((value) => setRecord({ run, value, error: null }))
      .catch((e: Error) => { if (e.name !== "AbortError") setRecord({ run, value: null, error: e.message }); });
    return () => controller.abort();
  }, [run, target]);

  const latest = useRef({ frame: target, count: timeline.data?.length ?? 0 });
  latest.current = { frame: target, count: timeline.data?.length ?? 0 };
  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => {
      const { frame: f, count } = latest.current;
      if (f + 1 >= count) { setPlaying(false); return; }
      write(parseHash().page, run, f + 1);
    }, 500);
    return () => clearInterval(id);
  }, [playing, run, write]);

  const loaded = record.run === run ? record.value : null;
  const value = useMemo<Selection>(() => ({
    page: route.page, run, frame: loaded?.frame ?? target, target, sequences: sequences.data, sequencesError: sequences.error, detail: detail.data, timeline: timeline.data,
    record: loaded, recordError: record.run === run ? record.error : null,
    sequence: sequences.data?.find((s) => s.runs.some((r) => r.id === run)) ?? null, playing, setPlaying,
    go: (page) => write(page, run, target), setRun: (next) => { setPlaying(false); write(route.page, next, 0); }, setFrame: (f) => write(route.page, run, f), reload: () => setRefresh((r) => r + 1),
  }), [route.page, run, target, loaded, sequences.data, sequences.error, detail.data, timeline.data, record, playing, write]);
  return <Context.Provider value={value}>{children}</Context.Provider>;
}
