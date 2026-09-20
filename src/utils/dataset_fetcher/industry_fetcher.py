"""Fetch raw daily A-share industry classification snapshots from Tushare.

The fetcher follows the industry-classification part of
``/opt/tiger/qyd/quant_llm/scripts/preprocess/run_fetch_all_ashares.py`` and
extends it into daily raw snapshots.  It pulls Shenwan industry metadata through
``index_member_all`` when available, falls back to ``index_classify`` +
``index_member``, then expands membership intervals into date-partitioned
parquet files:

```
<output_dir>/industry/year=YYYY/month=MM/YYYYMMDD.parquet
```

Progress is saved under ``<output_dir>/industry_fetch_progress.json`` so an
interrupted run can be resumed safely.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


DEFAULT_RAW_DATA_DIR = Path("/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/raw_data")
DEFAULT_HTTP_URL = os.environ.get("DEFAULT_HTTP_URL") or os.environ.get("TUSHARE_HTTP_URL") or "http://jiaoch.site"
PROGRESS_FILE_NAME = "industry_fetch_progress.json"

# THS 行业（stock_basic.industry）→ 申万 SW2021 L1 映射表。
# 用于补充申万指数未覆盖的股票（如北交所、部分新股）。
# 映射规则：对同时有 THS 行业和 SW L1 的股票，取该 THS 行业下占比最高的 SW L1。
THS_TO_SW_L1_MAP: dict[str, str] = {
    "IT设备": "计算机",
    "专用机械": "机械设备",
    "中成药": "医药生物",
    "乳制品": "食品饮料",
    "互联网": "传媒",
    "仓储物流": "交通运输",
    "供气供热": "公用事业",
    "保险": "非银金融",
    "元器件": "电子",
    "全国地产": "房地产",
    "公共交通": "交通运输",
    "公路": "交通运输",
    "其他商业": "商贸零售",
    "其他建材": "建筑材料",
    "农业综合": "农林牧渔",
    "农用机械": "机械设备",
    "农药化肥": "基础化工",
    "出版业": "传媒",
    "化学制药": "医药生物",
    "化工原料": "基础化工",
    "化工机械": "机械设备",
    "化纤": "基础化工",
    "区域地产": "房地产",
    "医疗保健": "医药生物",
    "医药商业": "医药生物",
    "半导体": "电子",
    "商品城": "商贸零售",
    "商贸代理": "商贸零售",
    "啤酒": "食品饮料",
    "园区开发": "房地产",
    "塑料": "基础化工",
    "多元金融": "非银金融",
    "家居用品": "轻工制造",
    "家用电器": "家用电器",
    "小金属": "有色金属",
    "工程机械": "机械设备",
    "广告包装": "轻工制造",
    "建筑工程": "建筑装饰",
    "影视音像": "传媒",
    "房产服务": "房地产",
    "批发业": "基础化工",
    "摩托车": "汽车",
    "文教休闲": "社会服务",
    "新型电力": "公用事业",
    "旅游景点": "社会服务",
    "旅游服务": "社会服务",
    "日用化工": "美容护理",
    "普钢": "钢铁",
    "服饰": "纺织服饰",
    "机场": "交通运输",
    "机床制造": "机械设备",
    "机械基件": "机械设备",
    "林业": "农林牧渔",
    "染料涂料": "基础化工",
    "橡胶": "基础化工",
    "水力发电": "公用事业",
    "水务": "环保",
    "水泥": "建筑材料",
    "水运": "交通运输",
    "汽车整车": "汽车",
    "汽车服务": "汽车",
    "汽车配件": "汽车",
    "渔业": "农林牧渔",
    "港口": "交通运输",
    "火力发电": "公用事业",
    "焦炭加工": "煤炭",
    "煤炭开采": "煤炭",
    "特种钢": "钢铁",
    "环境保护": "环保",
    "玻璃": "建筑材料",
    "生物制药": "医药生物",
    "电信运营": "通信",
    "电器仪表": "机械设备",
    "电器连锁": "商贸零售",
    "电气设备": "电力设备",
    "白酒": "食品饮料",
    "百货": "商贸零售",
    "石油加工": "石油石化",
    "石油开采": "石油石化",
    "石油贸易": "石油石化",
    "矿物制品": "有色金属",
    "种植业": "农林牧渔",
    "空运": "交通运输",
    "红黄酒": "食品饮料",
    "纺织": "纺织服饰",
    "纺织机械": "机械设备",
    "综合类": "综合",
    "航空": "国防军工",
    "船舶": "国防军工",
    "装修装饰": "建筑装饰",
    "证券": "非银金融",
    "超市连锁": "商贸零售",
    "路桥": "交通运输",
    "软件服务": "计算机",
    "软饮料": "食品饮料",
    "轻工机械": "机械设备",
    "运输设备": "机械设备",
    "通信设备": "通信",
    "造纸": "轻工制造",
    "酒店餐饮": "社会服务",
    "钢加工": "机械设备",
    "铁路": "交通运输",
    "铅锌": "有色金属",
    "铜": "有色金属",
    "铝": "有色金属",
    "银行": "银行",
    "陶瓷": "轻工制造",
    "食品": "食品饮料",
    "饲料": "农林牧渔",
    "黄金": "有色金属",
}


@dataclass(frozen=True)
class IndustryFetchSummary:
    """Summary returned after an industry-classification fetch run."""

    start_date: str
    end_date: str
    total_trade_dates: int
    fetched_trade_dates: int
    skipped_trade_dates: int
    failed_trade_dates: tuple[str, ...]
    rows_written: int
    output_dir: Path


class IndustryFetcher:
    """Fetch raw daily Shenwan industry classification snapshots from Tushare.

    Parameters
    ----------
    token
        Tushare token. If omitted, it is loaded from ``TUSHARE_TOKEN`` or
        ``~/.tushare_token``.
    output_dir
        Base raw-data directory. Industry files are written into its
        ``industry`` subdirectory. Defaults to
        ``/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/raw_data``.
    http_url
        Optional custom Tushare-compatible API endpoint, useful when using a
        proxy service.
    """

    INDUSTRY_SUBDIR = "industry"
    COMPLETED_PROGRESS_KEY = "industry_completed_dates"

    def __init__(
        self,
        token: str | None = None,
        output_dir: str | Path | None = None,
        http_url: str | None = DEFAULT_HTTP_URL,
        logger: logging.Logger | None = None,
    ) -> None:
        self.token = token or self._load_token()
        self.output_dir = Path(output_dir or DEFAULT_RAW_DATA_DIR)
        self.industry_dir = self.output_dir / self.INDUSTRY_SUBDIR
        self.progress_file = self.output_dir / PROGRESS_FILE_NAME
        self.http_url = http_url
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.pro = self._init_tushare_client()

    def fetch(
        self,
        start_date: str,
        end_date: str | None = None,
        level: str = "L1",
        src: str = "SW",
        sleep_time: float = 0.2,
        retry_wait_seconds: float = 2.0,
        max_retries: int = 3,
        force_fetch: bool = False,
        filter_ashares: bool = False,
        fill_missing_via_stock_basic: bool = True,
    ) -> IndustryFetchSummary:
        """Fetch daily industry classification between ``start_date`` and ``end_date``.

        Dates must use ``YYYYMMDD`` format. ``end_date`` defaults to today.
        The generated daily snapshot keeps constituents whose ``in_date`` is on
        or before the snapshot date and whose ``out_date`` is empty or after the
        snapshot date.  When ``force_fetch`` is false, dates already recorded in
        the progress file and already present on disk are skipped.

        When ``fill_missing_via_stock_basic`` is true (default), stocks not
        covered by the primary index-based source (e.g. SW2021 index_member does not cover
        BSE) are supplemented via ``stock_basic.industry`` (THS industry)
        mapped to the target level.
        """

        end_date = end_date or datetime.now().strftime("%Y%m%d")
        self._validate_date(start_date, "start_date")
        self._validate_date(end_date, "end_date")
        if start_date > end_date:
            raise ValueError(f"start_date must be <= end_date, got {start_date} > {end_date}")
        if max_retries <= 0:
            raise ValueError("max_retries must be positive")
        if sleep_time < 0:
            raise ValueError("sleep_time must be non-negative")
        if retry_wait_seconds < 0:
            raise ValueError("retry_wait_seconds must be non-negative")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.industry_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("开始拉取 raw industry：%s ~ %s，src=%s，level=%s", start_date, end_date, src, level)
        trade_dates = self.get_trade_dates(start_date, end_date, retry_wait_seconds=retry_wait_seconds)
        if not trade_dates:
            raise RuntimeError(f"未获取到交易日历：{start_date} ~ {end_date}")
        self.logger.info("交易日数量：%d", len(trade_dates))

        progress = self._load_progress()
        completed_dates = set(progress.get(self.COMPLETED_PROGRESS_KEY, []))
        pending_dates = (
            trade_dates
            if force_fetch
            else [d for d in trade_dates if d not in completed_dates or not self._date_output_file_exists(d)]
        )
        skipped_dates = 0 if force_fetch else len(trade_dates) - len(pending_dates)

        if not pending_dates:
            self.logger.info("industry 已全部拉取完成，无需重复拉取。")
            return IndustryFetchSummary(
                start_date=start_date,
                end_date=end_date,
                total_trade_dates=len(trade_dates),
                fetched_trade_dates=0,
                skipped_trade_dates=skipped_dates,
                failed_trade_dates=(),
                rows_written=0,
                output_dir=self.industry_dir,
            )

        industry_members = self.fetch_industry_members(
            level=level,
            src=src,
            sleep_time=sleep_time,
            retry_wait_seconds=retry_wait_seconds,
            max_retries=max_retries,
        )
        if industry_members.empty:
            raise RuntimeError(f"未获取到行业成分数据：src={src}, level={level}")

        if filter_ashares and "ts_code" in industry_members.columns:
            ts_codes_set = set(self.get_all_ashare_codes())
            industry_members = industry_members[industry_members["ts_code"].astype(str).isin(ts_codes_set)].copy()
            self.logger.info("按全量 A 股列表过滤后，行业成分记录数：%d", len(industry_members))

        if fill_missing_via_stock_basic and level == "L1":
            industry_members = self._fill_missing_via_stock_basic(
                industry_members=industry_members,
                src=src,
                level=level,
                sleep_time=sleep_time,
                retry_wait_seconds=retry_wait_seconds,
                max_retries=max_retries,
            )

        rows_written = 0
        fetched_dates: list[str] = []
        failed_dates: list[str] = []

        for index, trade_date in enumerate(pending_dates, start=1):
            try:
                daily_df = self.build_daily_snapshot(industry_members, trade_date)
                rows_written += self._save_one_trade_date(trade_date, daily_df)
                fetched_dates.append(trade_date)
                completed_dates.add(trade_date)
            except Exception as exc:  # pragma: no cover - mainly filesystem/parquet engine dependent
                failed_dates.append(trade_date)
                self.logger.warning("保存 industry %s 失败：%s", trade_date, exc)
                continue

            progress[self.COMPLETED_PROGRESS_KEY] = sorted(completed_dates)
            self._save_progress(progress)
            if index % 20 == 0 or index == len(pending_dates):
                self.logger.info(
                    "industry 进度：本次 %d/%d，累计完成 %d/%d，失败累计=%d，累计写入行数=%d",
                    index,
                    len(pending_dates),
                    len(completed_dates),
                    len(trade_dates),
                    len(failed_dates),
                    rows_written,
                )

        summary = IndustryFetchSummary(
            start_date=start_date,
            end_date=end_date,
            total_trade_dates=len(trade_dates),
            fetched_trade_dates=len(fetched_dates),
            skipped_trade_dates=skipped_dates,
            failed_trade_dates=tuple(failed_dates),
            rows_written=rows_written,
            output_dir=self.industry_dir,
        )
        self._log_summary(summary)
        return summary

    def fetch_industry_members(
        self,
        level: str = "L1",
        src: str = "SW",
        sleep_time: float = 0.2,
        retry_wait_seconds: float = 2.0,
        max_retries: int = 3,
    ) -> pd.DataFrame:
        """Fetch industry metadata and member intervals from Tushare.

        Uses ``index_classify`` + ``index_member`` per-industry fetch as the
        primary path, which gives full coverage (all SW2021 industries).
        Falls back to ``index_member_all`` if per-industry fetch fails.
        """

        industry_meta = self._fetch_industry_classify(
            level=level,
            src=src,
            retry_wait_seconds=retry_wait_seconds,
            max_retries=max_retries,
        )
        if not industry_meta.empty:
            self.logger.info("通过 index_classify 获取到行业数量：%d (src=%s, level=%s)", len(industry_meta), src, level)

            frames: list[pd.DataFrame] = []
            empty_count = 0
            failed_count = 0
            index_codes = industry_meta["index_code"].dropna().astype(str).unique()
            for index_code in index_codes:
                member_df = self._fetch_one_industry_member(
                    index_code=index_code,
                    sleep_time=sleep_time,
                    retry_wait_seconds=retry_wait_seconds,
                    max_retries=max_retries,
                )
                if member_df is not None and not member_df.empty:
                    frames.append(member_df)
                elif member_df is None:
                    failed_count += 1
                else:
                    empty_count += 1

            if frames:
                self.logger.info(
                    "通过 index_member 获取到非空行业数量：%d，空行业：%d，失败：%d",
                    len(frames),
                    empty_count,
                    failed_count,
                )
                members = pd.concat(frames, ignore_index=True)
                members = members.merge(
                    industry_meta,
                    on="index_code",
                    how="left",
                    suffixes=("", "_classify"),
                )
                return self._normalize_industry_members(members, src=src, level=level)

        self.logger.warning("index_classify + index_member 路径未返回数据，回退到 index_member_all。")

        member_all = self._fetch_index_member_all(
            level=level,
            src=src,
            retry_wait_seconds=retry_wait_seconds,
            max_retries=max_retries,
        )
        if not member_all.empty:
            self.logger.info("通过 index_member_all 获取到行业成分记录数：%d", len(member_all))
            return member_all

        return pd.DataFrame()

    def _fetch_index_member_all(
        self,
        level: str,
        src: str,
        retry_wait_seconds: float,
        max_retries: int,
    ) -> pd.DataFrame:
        """Fetch SW industry constituents through Tushare ``index_member_all``."""

        if not hasattr(self.pro, "index_member_all"):
            return pd.DataFrame()

        fields = (
            "l1_code,l1_name,l2_code,l2_name,l3_code,l3_name,"
            "ts_code,name,in_date,out_date,is_new"
        )
        last_empty: pd.DataFrame | None = None
        for params in ({}, {"is_new": "Y"}):
            for retry_idx in range(1, max_retries + 1):
                try:
                    df = self.pro.index_member_all(fields=fields, **params)
                    if df is None:
                        df = pd.DataFrame()
                    if df.empty:
                        last_empty = df
                        break
                    return self._normalize_index_member_all(df, src=src, level=level)
                except Exception as exc:  # pragma: no cover - depends on remote API
                    self.logger.warning(
                        "获取 index_member_all params=%s 失败（第 %d/%d 次）：%s",
                        params,
                        retry_idx,
                        max_retries,
                        exc,
                    )
                    if retry_idx < max_retries:
                        time.sleep(retry_wait_seconds * retry_idx)

        return last_empty if last_empty is not None else pd.DataFrame()

    @staticmethod
    def _normalize_index_member_all(df: pd.DataFrame, src: str, level: str) -> pd.DataFrame:
        level_to_cols = {
            "L1": ("l1_code", "l1_name"),
            "L2": ("l2_code", "l2_name"),
            "L3": ("l3_code", "l3_name"),
        }
        code_col, name_col = level_to_cols.get(level, ("l1_code", "l1_name"))
        if code_col not in df.columns or name_col not in df.columns:
            raise ValueError(
                f"index_member_all response does not contain required columns for {level}: "
                f"{code_col}, {name_col}; columns={list(df.columns)}"
            )
        if "ts_code" not in df.columns:
            raise ValueError(f"index_member_all response does not contain required column: ts_code; columns={list(df.columns)}")

        normalized = df.copy()
        normalized["industry_code"] = normalized[code_col]
        normalized["industry"] = normalized[name_col]
        normalized["level"] = level
        normalized["src"] = src
        if "in_date" not in normalized.columns:
            normalized["in_date"] = "00000000"
        if "out_date" not in normalized.columns:
            normalized["out_date"] = ""

        normalized = normalized[normalized["industry_code"].notna() & normalized["ts_code"].notna()].copy()
        first_cols = [
            "ts_code",
            "name",
            "industry_code",
            "industry",
            "level",
            "src",
            "in_date",
            "out_date",
            "is_new",
        ]
        ordered_cols = [col for col in first_cols if col in normalized.columns]
        ordered_cols.extend(col for col in normalized.columns if col not in ordered_cols)
        return normalized[ordered_cols]

    def _fill_missing_via_stock_basic(
        self,
        industry_members: pd.DataFrame,
        src: str,
        level: str,
        sleep_time: float,
        retry_wait_seconds: float,
        max_retries: int,
    ) -> pd.DataFrame:
        """Supplement industry data for stocks missing from index_member.

        Uses ``stock_basic.industry`` (THS industry classification) mapped to
        SW L1 via :data:`THS_TO_SW_L1_MAP`.  Covers two gap scenarios:

        1. **Pre-SW-entry gap**: Newly listed stocks that have not yet been
           added to SW indices (e.g. IPO in late Dec, SW entry in early Jan).
        2. **Post-SW-exit gap**: Stocks that have been removed from SW indices
           (e.g. out_date=20251231 means no longer in SW on 20251231) but are
           still listed and tradable.

        For each stock not already covered by SW on every date, a THS-mapped
        record is added with ``in_date = list_date`` and ``out_date = ""``
        (active indefinitely).  In ``build_daily_snapshot``, when both a SW
        record and a THS_MAPPED record exist for the same stock on the same
        date, the SW record takes priority (via the sort order).
        """

        if industry_members.empty:
            return industry_members

        existing_codes = set(industry_members["ts_code"].astype(str).unique())
        self.logger.info("stock_basic 补充前：已有 %d 只股票的行业数据", len(existing_codes))

        frames: list[pd.DataFrame] = []
        for list_status in ("L", "D", "P"):
            for retry_idx in range(1, max_retries + 1):
                try:
                    df = self.pro.stock_basic(
                        list_status=list_status,
                        fields="ts_code,name,industry,list_date,market",
                    )
                    if df is not None and not df.empty:
                        frames.append(df)
                    break
                except Exception as exc:
                    self.logger.warning(
                        "获取 stock_basic list_status=%s 失败（第 %d/%d 次）：%s",
                        list_status, retry_idx, max_retries, exc,
                    )
                    if retry_idx < max_retries:
                        time.sleep(retry_wait_seconds * retry_idx)
            time.sleep(sleep_time)

        if not frames:
            self.logger.warning("stock_basic 未返回数据，跳过补充。")
            return industry_members

        all_stocks = pd.concat(frames, ignore_index=True)
        all_stocks["ts_code"] = all_stocks["ts_code"].astype(str)
        all_stocks = all_stocks.drop_duplicates(subset=["ts_code"], keep="first")

        # Only supplement stocks that have a THS industry
        stocks_with_industry = all_stocks[all_stocks["industry"].notna()].copy()

        if stocks_with_industry.empty:
            self.logger.info("stock_basic 补充：无有行业信息的股票。")
            return industry_members

        stocks_with_industry["sw_l1"] = stocks_with_industry["industry"].map(THS_TO_SW_L1_MAP)
        unmapped = stocks_with_industry[stocks_with_industry["sw_l1"].isna()]
        if not unmapped.empty:
            self.logger.warning(
                "stock_basic 补充：%d 只股票的 THS 行业无法映射到 SW L1，已跳过。未映射行业：%s",
                len(unmapped),
                sorted(unmapped["industry"].unique().tolist()),
            )
            stocks_with_industry = stocks_with_industry[stocks_with_industry["sw_l1"].notna()].copy()

        if stocks_with_industry.empty:
            self.logger.info("stock_basic 补充：无可映射股票。")
            return industry_members

        supplement = pd.DataFrame({
            "ts_code": stocks_with_industry["ts_code"].values,
            "name": stocks_with_industry["name"].values,
            "industry_code": "THS_MAPPED",
            "industry": stocks_with_industry["sw_l1"].values,
            "level": level,
            "src": src,
            "in_date": stocks_with_industry["list_date"].fillna("00000000").astype(str).str.replace("-", "", regex=False).values,
            "out_date": "",
            "is_new": "N",
            "industry_code_raw": stocks_with_industry["industry"].values,
            "is_pub": "1",
            "parent_code": "",
        })

        result = pd.concat([industry_members, supplement], ignore_index=True)
        self.logger.info(
            "stock_basic 补充完成：新增 %d 只股票的 THS→SW L1 映射行业数据，总计 %d 只",
            len(supplement),
            result["ts_code"].nunique(),
        )
        return result

    def build_daily_snapshot(self, industry_members: pd.DataFrame, trade_date: str) -> pd.DataFrame:
        """Build one daily stock-to-industry snapshot from member intervals."""

        if industry_members.empty:
            return industry_members.copy()

        snapshot = industry_members.copy()
        in_date = snapshot["in_date"].fillna("00000000").astype(str).str.replace("-", "", regex=False)
        out_date = snapshot["out_date"].fillna("").astype(str).str.replace("-", "", regex=False)
        out_date = out_date.mask(out_date.isin(("", "nan", "NaT", "None")), "99991231")

        active_mask = (in_date <= trade_date) & (out_date > trade_date)
        snapshot = snapshot[active_mask].copy()
        snapshot.insert(0, "trade_date", trade_date)

        # SW 行业记录优先于 THS 映射记录。当同一股票同一天同时存在 SW 和 THS_MAPPED 时，
        # 保留 SW 记录（industry_code 为真实指数代码，排在 THS_MAPPED 前面）。
        if "is_new" in snapshot.columns:
            snapshot = snapshot.sort_values(
                ["ts_code", "is_new", "in_date", "industry_code"],
                ascending=[True, False, False, True],
            )
        else:
            snapshot = snapshot.sort_values(
                ["ts_code", "in_date", "industry_code"],
                ascending=[True, False, True],
            )
        snapshot = snapshot.drop_duplicates(subset=["trade_date", "ts_code"], keep="first")
        return snapshot.reset_index(drop=True)

    def get_trade_dates(self, start_date: str, end_date: str, retry_wait_seconds: float = 2.0) -> list[str]:
        """Return open SSE trading dates in ascending order."""

        for retry_idx in range(1, 4):
            try:
                df = self.pro.trade_cal(exchange="SSE", start_date=start_date, end_date=end_date, is_open="1")
                if df is None or df.empty:
                    return []
                return sorted(df["cal_date"].astype(str).tolist())
            except Exception as exc:  # pragma: no cover - depends on remote API
                self.logger.warning(
                    "获取交易日历失败（第 %d/3 次）：%s",
                    retry_idx,
                    exc,
                )
                if retry_idx < 3:
                    time.sleep(retry_wait_seconds * retry_idx)

        fallback_dates = self._calendar_dates(start_date, end_date)
        self.logger.warning(
            "交易日历接口连续失败，将退化为逐自然日生成 industry 快照，日期数=%d。",
            len(fallback_dates),
        )
        return fallback_dates

    @staticmethod
    def _calendar_dates(start_date: str, end_date: str) -> list[str]:
        start = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")
        dates: list[str] = []
        current = start
        while current <= end:
            dates.append(current.strftime("%Y%m%d"))
            current += timedelta(days=1)
        return dates

    def get_all_ashare_codes(self) -> list[str]:
        """Return all A-share ts_code values, including delisted/suspended ones."""

        frames: list[pd.DataFrame] = []
        for list_status in ("L", "D", "P"):
            try:
                df = self.pro.stock_basic(list_status=list_status)
                if df is not None and not df.empty:
                    frames.append(df)
            except Exception as exc:  # pragma: no cover - depends on remote API
                self.logger.warning("获取 list_status=%s 股票列表失败：%s", list_status, exc)
            time.sleep(0.2)

        if not frames:
            raise RuntimeError("无法获取 A 股股票列表")

        stocks = pd.concat(frames, ignore_index=True)
        return sorted(stocks["ts_code"].dropna().astype(str).unique().tolist())

    def _fetch_industry_classify(
        self,
        level: str,
        src: str,
        retry_wait_seconds: float,
        max_retries: int,
    ) -> pd.DataFrame:
        fields = "index_code,industry_name,level,industry_code,is_pub,parent_code,src"
        for retry_idx in range(1, max_retries + 1):
            try:
                df = self.pro.index_classify(level=level, src=src, fields=fields)
                if df is None:
                    return pd.DataFrame()
                if "index_code" not in df.columns:
                    preview = df.head(3).to_dict(orient="records") if hasattr(df, "head") else repr(df)
                    raise ValueError(
                        "index_classify response does not contain required column: index_code; "
                        f"shape={getattr(df, 'shape', None)}, columns={list(getattr(df, 'columns', []))}, "
                        f"preview={preview}"
                    )
                return df
            except Exception as exc:  # pragma: no cover - depends on remote API
                self.logger.warning(
                    "获取 index_classify src=%s level=%s 失败（第 %d/%d 次）：%s",
                    src,
                    level,
                    retry_idx,
                    max_retries,
                    exc,
                )
                if retry_idx < max_retries:
                    time.sleep(retry_wait_seconds * retry_idx)

        raise RuntimeError(f"获取 index_classify 最终失败：src={src}, level={level}")

    def _fetch_one_industry_member(
        self,
        index_code: str,
        sleep_time: float,
        retry_wait_seconds: float,
        max_retries: int,
    ) -> pd.DataFrame | None:
        for retry_idx in range(1, max_retries + 1):
            try:
                df = self.pro.index_member(index_code=index_code)
                if df is None:
                    df = pd.DataFrame()
                time.sleep(sleep_time)
                return df
            except Exception as exc:  # pragma: no cover - depends on remote API
                self.logger.warning(
                    "获取 index_member %s 失败（第 %d/%d 次）：%s",
                    index_code,
                    retry_idx,
                    max_retries,
                    exc,
                )
                if retry_idx < max_retries:
                    time.sleep(retry_wait_seconds * retry_idx)

        self.logger.error("获取 index_member %s 最终失败，跳过该行业。", index_code)
        return None

    @staticmethod
    def _normalize_industry_members(df: pd.DataFrame, src: str, level: str) -> pd.DataFrame:
        normalized = df.copy()
        if "index_code" in normalized.columns and "industry_code" in normalized.columns:
            normalized = normalized.rename(columns={"industry_code": "industry_code_raw"})

        rename_map = {
            "con_code": "ts_code",
            "con_name": "name",
            "index_code": "industry_code",
            "industry_name": "industry",
        }
        normalized = normalized.rename(columns={k: v for k, v in rename_map.items() if k in normalized.columns})

        if "industry_code" not in normalized.columns:
            raise ValueError("index_member/index_classify response does not contain required industry code")
        if "ts_code" not in normalized.columns:
            raise ValueError("index_member response does not contain required constituent code column: con_code")

        if "industry" not in normalized.columns and "index_name" in normalized.columns:
            normalized["industry"] = normalized["index_name"]
        if "src" not in normalized.columns:
            normalized["src"] = src
        if "level" not in normalized.columns:
            normalized["level"] = level
        if "in_date" not in normalized.columns:
            normalized["in_date"] = "00000000"
        if "out_date" not in normalized.columns:
            normalized["out_date"] = ""

        first_cols = [
            "ts_code",
            "name",
            "industry_code",
            "industry",
            "level",
            "src",
            "in_date",
            "out_date",
            "is_new",
        ]
        ordered_cols = [col for col in first_cols if col in normalized.columns]
        ordered_cols.extend(col for col in normalized.columns if col not in ordered_cols)
        return normalized[ordered_cols]

    def _save_one_trade_date(self, trade_date: str, df: pd.DataFrame) -> int:
        partition_dir = self.industry_dir / f"year={trade_date[:4]}" / f"month={trade_date[4:6]}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        file_path = partition_dir / f"{trade_date}.parquet"
        df.to_parquet(file_path, compression="snappy", index=False)
        return len(df)

    def _date_output_file_exists(self, trade_date: str) -> bool:
        return (
            self.industry_dir
            / f"year={trade_date[:4]}"
            / f"month={trade_date[4:6]}"
            / f"{trade_date}.parquet"
        ).exists()

    def _init_tushare_client(self):
        try:
            import tushare as ts
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("未安装 tushare，请先执行：pip install tushare") from exc

        ts.set_token(self.token)
        pro = ts.pro_api(self.token)
        if self.http_url:
            pro._DataApi__http_url = self.http_url
        pro._DataApi__token = self.token
        return pro

    @staticmethod
    def _load_token() -> str:
        token = os.environ.get("TUSHARE_TOKEN")
        if token:
            return token.strip()

        token_file = Path.home() / ".tushare_token"
        if token_file.exists():
            token = token_file.read_text(encoding="utf-8").strip()
            if token:
                return token

        raise ValueError("请通过 --token、环境变量 TUSHARE_TOKEN 或 ~/.tushare_token 提供 Tushare Token")

    @staticmethod
    def _validate_date(value: str, name: str) -> None:
        try:
            datetime.strptime(value, "%Y%m%d")
        except ValueError as exc:
            raise ValueError(f"{name} must use YYYYMMDD format, got {value!r}") from exc

    def _load_progress(self) -> dict[str, list[str]]:
        if not self.progress_file.exists():
            return {}
        with self.progress_file.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _save_progress(self, progress: dict[str, list[str]]) -> None:
        with self.progress_file.open("w", encoding="utf-8") as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)

    def _log_summary(self, summary: IndustryFetchSummary) -> None:
        self.logger.info("=" * 60)
        self.logger.info("industry 拉取完成")
        self.logger.info("时间范围：%s ~ %s", summary.start_date, summary.end_date)
        self.logger.info("交易日总数：%d", summary.total_trade_dates)
        self.logger.info("本次成功生成交易日：%d", summary.fetched_trade_dates)
        self.logger.info("本次跳过交易日：%d", summary.skipped_trade_dates)
        self.logger.info("失败交易日：%s", list(summary.failed_trade_dates))
        self.logger.info("写入行数：%d", summary.rows_written)
        self.logger.info("输出目录：%s", summary.output_dir)
        self.logger.info("=" * 60)


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("industry_fetcher")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="拉取 Tushare 原始每日行业分类 industry")
    parser.add_argument("--token", type=str, default=None, help="Tushare Token；默认读取 TUSHARE_TOKEN 或 ~/.tushare_token")
    parser.add_argument("--start_date", type=str, required=True, help="开始日期，格式 YYYYMMDD")
    parser.add_argument("--end_date", type=str, default=None, help="结束日期，格式 YYYYMMDD；默认今天")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_RAW_DATA_DIR),
        help=f"raw_data 根目录；默认 {DEFAULT_RAW_DATA_DIR}",
    )
    parser.add_argument("--http_url", type=str, default=DEFAULT_HTTP_URL, help="可选 Tushare 兼容代理地址")
    parser.add_argument("--level", type=str, default="L1", choices=["L1", "L2", "L3"], help="行业级别；默认申万一级 L1")
    parser.add_argument("--src", type=str, default="SW2021", help="行业分类来源；默认 SW2021（申万2021版）")
    parser.add_argument("--sleep_time", type=float, default=0.2, help="每次成功请求后的等待秒数")
    parser.add_argument(
        "--retry_wait_seconds",
        type=float,
        default=2.0,
        help="请求失败后重试的基础等待秒数；实际等待为 retry_wait_seconds * retry_idx",
    )
    parser.add_argument("--max_retries", type=int, default=3, help="单个接口最大重试次数")
    parser.add_argument("--force_fetch", action="store_true", help="忽略进度文件，强制重新拉取并覆盖同名 parquet")
    parser.add_argument(
        "--filter_ashares",
        action="store_true",
        help="额外按全量 A 股股票列表过滤 index_member 返回结果；默认不启用",
    )
    parser.add_argument(
        "--no_fill_missing",
        action="store_true",
        help="禁用 stock_basic THS 行业补充；默认启用，用于覆盖申万指数未覆盖的股票（如北交所）",
    )
    parser.add_argument("--log_level", type=str, default=os.environ.get("LOG_LEVEL", "INFO"), help="日志级别")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logger = setup_logging(args.log_level)
    fetcher = IndustryFetcher(
        token=args.token,
        output_dir=args.output_dir,
        http_url=args.http_url,
        logger=logger,
    )
    summary = fetcher.fetch(
        start_date=args.start_date,
        end_date=args.end_date,
        level=args.level,
        src=args.src,
        sleep_time=args.sleep_time,
        retry_wait_seconds=args.retry_wait_seconds,
        max_retries=args.max_retries,
        force_fetch=args.force_fetch,
        filter_ashares=args.filter_ashares,
        fill_missing_via_stock_basic=not args.no_fill_missing,
    )
    return 1 if summary.failed_trade_dates else 0


if __name__ == "__main__":
    raise SystemExit(main())
