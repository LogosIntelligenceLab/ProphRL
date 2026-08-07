import time, threading, queue, uuid
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor

import ray
from swift.plugin import InferStats
from swift.llm.infer.protocol import RequestConfig, InferRequest
from swift.llm.infer.infer_engine.infer_client import InferClient

@dataclass
class _Job:
    key: str
    req: InferRequest

@ray.remote(num_cpus=0)
class RMGateway:
    def __init__(
        self,
        endpoints: List[Tuple[str, int]],
        max_batch_size: int = 256,
        max_wait_ms: int = 10,
        default_cfg: Optional[dict] = None,
        round_robin: bool = True,
        queue_maxsize: int = 10000,
        concurrency: int = None,      # 线程池并发度（建议 = 端点数）
        max_inflight: int = None,     # 同时在飞的批数（建议 = 端点数）
    ):
        self.endpoints = endpoints
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.default_cfg = default_cfg or {"temperature": 0.0, "max_tokens": 32, "stream": False}
        self.round_robin = round_robin

        # === 客户端缓存 & 端点健康状态 ===
        self._clients = [InferClient(host=h, port=p) for (h, p) in endpoints]
        self._alive = [True] * len(self._clients)
        self._idx = -1
        self._alive_lock = threading.Lock()

        self.metric = InferStats()
        self.q: "queue.Queue[_Job]" = queue.Queue(maxsize=queue_maxsize)
        self.results: Dict[str, object] = {}
        self._res_lock = threading.Lock()

        # 并发控制
        if concurrency is None:
            concurrency = max(1, len(self._clients))
        if max_inflight is None:
            max_inflight = concurrency
        self._executor = ThreadPoolExecutor(max_workers=concurrency)
        self._inflight_sem = threading.Semaphore(max_inflight)
        self._inflight_lock = threading.Lock()
        self._inflight_count = 0

        self._stop = threading.Event()
        self._t = threading.Thread(target=self._flush_loop, daemon=True)
        self._t.start()

    # ============ 对外 API ============
    def submit(self, req: InferRequest) -> str:
        k = uuid.uuid4().hex
        self.q.put(_Job(key=k, req=req), block=True)
        return k

    def submit_many(self, reqs: List[InferRequest]) -> List[str]:
        keys = []
        for r in reqs:
            k = uuid.uuid4().hex
            self.q.put(_Job(key=k, req=r), block=True)
            keys.append(k)
        return keys

    def get_result_blocking(self, key: str, timeout_s: Optional[float] = None):
        start = time.time()
        while True:
            with self._res_lock:
                res = self.results.pop(key, None)
            if res is not None:
                if isinstance(res, Exception):
                    raise res
                return res
            if timeout_s is not None and time.time() - start > timeout_s:
                raise TimeoutError(f"timeout waiting result for key={key}")
            time.sleep(0.002)

    def get_many_blocking(self, keys: List[str], timeout_s: Optional[float] = None):
        start = time.time()
        results = [None] * len(keys)
        remaining = set(range(len(keys)))
        while remaining:
            with self._res_lock:
                for i in list(remaining):
                    k = keys[i]
                    res = self.results.pop(k, None)
                    if res is not None:
                        results[i] = res
                        remaining.remove(i)
            if remaining:
                if timeout_s is not None and time.time() - start > timeout_s:
                    missing = [keys[i] for i in sorted(remaining)]
                    raise TimeoutError(f"timeout waiting keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
                time.sleep(0.002)
        # 如需把批中任一异常直接抛出（便于上层重试批次），保留下面这段
        for r in results:
            if isinstance(r, Exception):
                raise r
        return results

    def stop(self):
        self._stop.set()
        self._t.join(timeout=1.0)
        self._executor.shutdown(wait=False, cancel_futures=True)

    # ============ 内部 ============
    def _pick_client_idx(self) -> int:
        with self._alive_lock:
            n = len(self._clients)
            if n == 1 or not self.round_robin:
                return 0 if self._alive[0] else 0
            for _ in range(n):
                self._idx = (self._idx + 1) % n
                if self._alive[self._idx]:
                    return self._idx
            return 0  # 全挂时仍返回0，后续会写错误给结果

    def _mark_dead(self, idx: int):
        with self._alive_lock:
            self._alive[idx] = False

    def _mark_alive(self, idx: int):
        with self._alive_lock:
            self._alive[idx] = True

    def _dispatch_batch_async(self, jobs: List[_Job], serve_timeout_s: float = 120.0, retry: int = 1):
        client_idx = self._pick_client_idx()
        client = self._clients[client_idx]
        cfg = RequestConfig(**self.default_cfg)
        infer_reqs = [j.req for j in jobs]
        batch_keys = [j.key for j in jobs]
        done_flag = threading.Event()

        def _write_results(obj):
            with self._res_lock:
                if isinstance(obj, list):
                    for k, v in zip(batch_keys, obj):
                        if k not in self.results:
                            self.results[k] = v
                else:
                    for k in batch_keys:
                        if k not in self.results:
                            self.results[k] = obj

        def _work_once(c_idx, clin):
            t0 = time.time()
            try:
                print(f"[GW] dispatch endpoint={c_idx} size={len(jobs)}")
                outs = clin.infer(infer_reqs, request_config=cfg, metrics=[self.metric])
                self._mark_alive(c_idx)
                _write_results(outs)
                print(f"[GW] done     endpoint={c_idx} size={len(jobs)} elapsed={time.time()-t0:.2f}s")
                return True
            except Exception as e:
                self._mark_dead(c_idx)
                _write_results(e)
                print(f"[GW] ERROR    endpoint={c_idx} size={len(jobs)} exc={e}")
                return False

        def _watchdog():
            if not done_flag.wait(timeout=serve_timeout_s):
                # 只报警，不覆盖结果，避免把慢批直接判死
                print(f"[GW] WARN watchdog endpoint={client_idx} size={len(jobs)} >{serve_timeout_s}s still running")

        def _work():
            try:
                wd = threading.Thread(target=_watchdog, daemon=True)
                wd.start()
                ok = _work_once(client_idx, client)
                if (not ok) and retry > 0:
                    alt_idx = self._pick_client_idx()
                    alt_client = self._clients[alt_idx]
                    _work_once(alt_idx, alt_client)
            finally:
                done_flag.set()
                with self._inflight_lock:
                    self._inflight_count -= 1
                self._inflight_sem.release()

        with self._inflight_lock:
            self._inflight_count += 1
        self._executor.submit(_work)

    def _flush_loop(self):
        pending: List[_Job] = []
        last_ts = time.time()
        max_wait_s = self.max_wait_ms / 1000.0

        while not self._stop.is_set():
            # A) 无阻塞捞活（把队列里已有的尽量拿完）
            while True:
                try:
                    pending.append(self.q.get_nowait())
                except queue.Empty:
                    break

            # B) 连续派发：只要 (pending够/窗口到期) 且还有并发槽位，就一直发
            dispatched = False
            while pending and (len(pending) >= self.max_batch_size or (time.time() - last_ts) >= max_wait_s):
                # 尝试把可用的并发槽位一次性塞满
                launched = 0
                while pending and self._inflight_sem.acquire(blocking=False):
                    batch = pending[:self.max_batch_size] if len(pending) >= self.max_batch_size else pending[:]
                    del pending[:len(batch)]
                    self._dispatch_batch_async(batch)
                    launched += 1
                    last_ts = time.time()
                    # 如果下一轮 pending 还够 & 还有槽位，会继续 launch
                    if len(pending) < self.max_batch_size and (time.time() - last_ts) < max_wait_s:
                        break
                if launched == 0:
                    break
                dispatched = True

            if dispatched:
                # 刚发过批，回到循环顶部再看看是否还能继续发
                continue

            # C) 发不出去时：短暂阻塞等新活或窗口到期
            timeout = max(max_wait_s - (time.time() - last_ts), 0.0)
            try:
                job = self.q.get(timeout=timeout if not pending else 0.0)
                pending.append(job)
            except queue.Empty:
                # 窗口到期但样本不足也要发出“最后一小批”，避免饿死 GPU
                if pending and self._inflight_sem.acquire(blocking=False):
                    batch = pending[:]
                    pending.clear()
                    self._dispatch_batch_async(batch)
                    last_ts = time.time()
