import { useEffect, useState } from "react";
import { get } from "./api";

export interface Load<T> { data: T | null; error: string | null; loading: boolean }

/** Fetch by path. A result is only ever returned for the path it came from, so nothing stale shows as current. */
export function useApi<T>(path: string | null, refresh = 0): Load<T> {
  const [state, setState] = useState<Load<T> & { path: string | null }>({ path: null, data: null, error: null, loading: false });
  useEffect(() => {
    if (path === null) return;
    const controller = new AbortController();
    setState((s) => (s.path === path ? { ...s, loading: true } : { path, data: null, error: null, loading: true }));
    get<T>(path, controller.signal)
      .then((data) => setState({ path, data, error: null, loading: false }))
      .catch((e: Error) => { if (e.name !== "AbortError") setState({ path, data: null, error: e.message, loading: false }); });
    return () => controller.abort();
  }, [path, refresh]);
  if (state.path !== path) return { data: null, error: null, loading: path !== null };
  return state;
}
