# ============================================================
# Common deps
# ============================================================
suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(lubridate)
})

# ============================================================
# @concept: KDPI
# @fn: compute_kdpi
# @data_kind: transplant               # 主数据来自移植文件
# @needs: data/transplant.csv, data/KDRI_Scale_Fac.csv, data/KDPI_Mapping_2023.csv
# 说明：
# - 读取移植数据与两张映射表，计算 KDRI 与 KDPI 百分位
# - 需要移植表中存在：DON_AGE, DON_HGT_CM, DON_WGT_KG, DON_RACE, DON_HTN,
#   DON_HIST_DIAB, DON_CAD_DON_COD, DON_CREAT, DON_HIGH_CREAT, DON_ANTI_HCV,
#   DON_NON_HR_BEAT, CAN_LISTING_DT 等
# ============================================================
compute_kdpi <- function(
  transplant_csv,
  kdri_scale_csv = "data/KDRI_Scale_Fac.csv",
  kdpi_map_csv   = "data/KDPI_Mapping_2023.csv",
  out_csv        = "out_kdpi.csv"
) {
  tx_ki   <- read_csv(transplant_csv, show_col_types = FALSE)
  KDPI_sca <- read_csv(kdri_scale_csv, show_col_types = FALSE)
  KDPI_map <- read_csv(kdpi_map_csv, show_col_types = FALSE)

  # 你的原始线性预测器 + unscaled KDRI
  tx_kdpi <- tx_ki %>%
    mutate(
      KDRI_x = 0.0128*(DON_AGE-40) -
               0.0194*(DON_AGE-18)*(DON_AGE<18) +
               0.0107*(DON_AGE-50)*(DON_AGE>50) -
               0.0464*(DON_HGT_CM-170)/10 -
               0.0199*(DON_WGT_KG-80)/5 +
               0.179*(DON_RACE==16) +
               0.126*(DON_HTN==1) +
               0.13 *(DON_HIST_DIAB %in% c(2,3,4,5)) +
               0.0881*(DON_CAD_DON_COD==2) +
               0.22 *(DON_CREAT-1) -
               0.209*(DON_CREAT-1.5)*(DON_HIGH_CREAT==1) +
               0.24 *(DON_ANTI_HCV=='P') +
               0.133*(DON_NON_HR_BEAT=='Y'),
      KDRI_unsca = exp(KDRI_x),
      Year = as.numeric(substr(CAN_LISTING_DT, 1, 4))
    )

  # 年度缩放 + KDPI 百分位映射
  # 用向量化 join 代替 for-loop
  tx_kdpi <- tx_kdpi %>%
    left_join(KDPI_sca %>% select(Year, Scale_Fac), by = "Year") %>%
    mutate(KDRI = KDRI_unsca / Scale_Fac) %>%
    left_join(
      KDPI_map %>% select(Year, low, up, KDPI),
      by = "Year",
      relationship = "many-to-many"
    ) %>%
    group_by(rowid = row_number()) %>%
    mutate(
      KDPI = KDPI[which(KDRI <= up & KDRI > low)][1]
    ) %>%
    ungroup() %>%
    select(-rowid, -low, -up, -Scale_Fac)

  write.csv(tx_kdpi, out_csv, row.names = FALSE)
  invisible(tx_kdpi)
}

# ============================================================
# @concept: eGFR
# @fn: compute_egfr
# @data_kind: transplant_followup
# @needs: data/transplant.csv, data/followup.csv
# 说明：
# - 以 CKD-EPI 2021 公式计算移植前（或最近一次）eGFR
# - 需要字段：CAN_GENDER, REC_CREAT, REC_AGE_AT_TX
# ============================================================
compute_egfr <- function(
  transplant_csv,
  followup_csv,
  out_csv = "out_egfr.csv"
) {
  tx <- read_csv(transplant_csv, show_col_types = FALSE)
  fu <- read_csv(followup_csv, show_col_types = FALSE)

  # 这里示例直接在 transplant 表内计算（若需要，可与随访表对齐最近一次肌酐）
  tx <- tx %>%
    mutate(
      REC_eGFR = ifelse(
        CAN_GENDER == 'M',
        142 * pmin(REC_CREAT/0.9, 1)^(-0.302) * pmax(REC_CREAT/0.9, 1)^(-1.2) * 0.9938^(REC_AGE_AT_TX),
        142 * pmin(REC_CREAT/0.7, 1)^(-0.241) * pmax(REC_CREAT/0.7, 1)^(-1.2) * 0.9938^(REC_AGE_AT_TX) * 1.012
      )
    )

  write.csv(tx, out_csv, row.names = FALSE)
  invisible(tx)
}

# ============================================================
# @concept: Transplant Rate
# @fn: compute_transplant_rate
# @data_kind: candidate
# @needs: data/candidate.csv
# 说明：
# - 借鉴你提供的月度“人月”分母与事件计数逻辑，计算各类移植率（总、尸体、活体）
# - 需要字段：CAN_LISTING_DT, CAN_REM_DT, CAN_REM_CD, WL_ORG, CAN_AGE_AT_LISTING
# ============================================================
compute_transplant_rate <- function(
  candidate_csv,
  out_csv = "out_transplant_rate.csv"
) {
  cand_kipa <- read_csv(candidate_csv, show_col_types = FALSE)

  dt <- cand_kipa %>%
    filter(WL_ORG == 'KI',
           is.na(CAN_REM_DT) | as.numeric(substr(CAN_REM_DT,1,4)) >= 2010,
           !is.na(CAN_LISTING_DT)) %>%
    mutate(
      start_month = pmax(as.numeric(substr(CAN_LISTING_DT,6,7)) + 12*(as.numeric(substr(CAN_LISTING_DT,1,4)) - 2010), 0),
      end_month   = as.numeric(substr(CAN_REM_DT,6,7)) + 12*(as.numeric(substr(CAN_REM_DT,1,4)) - 2010),
      day_start_month = as.numeric(substr(CAN_LISTING_DT,9,10)),
      day_end_month   = as.numeric(substr(CAN_REM_DT,9,10))
    )

  max_m <- suppressWarnings(max(dt$end_month, na.rm = TRUE))
  if (!is.finite(max_m)) max_m <- 1
  Txp_rate_plot_dt <- data.frame(month = 1:max_m)

  # 向量化近似（按你的循环逻辑）
  calc_days <- function(i) {
    if (i %% 12 %in% c(1,3,5,7,8,10,0)) return(31)
    if (i %% 12 == 2 && (i %/% 12) %% 4 == 0) return(29)
    if (i %% 12 == 2) return(28)
    30
  }
  days_vec <- vapply(Txp_rate_plot_dt$month, calc_days, numeric(1))

  out <- lapply(seq_len(nrow(Txp_rate_plot_dt)-1), function(i){
    days_at_month <- days_vec[i]
    tmp <- dt %>%
      filter(start_month <= i, is.na(end_month) | end_month >= i) %>%
      mutate(person_month = case_when(
        is.na(end_month)                ~ 1,
        start_month == end_month       ~ (day_end_month - day_start_month)/days_at_month,
        start_month == i               ~ (days_at_month + 1 - day_start_month)/days_at_month,
        end_month == i                 ~ day_end_month/days_at_month,
        TRUE                           ~ 1
      ))

    tibble(
      month = i,
      Txp_count            = nrow(filter(dt, CAN_REM_CD %in% c(4,15,18,19), end_month == i)),
      Deceased_Txp_count   = nrow(filter(dt, CAN_REM_CD %in% c(4), end_month == i)),
      Txp_free_death_count = nrow(filter(dt, CAN_REM_CD %in% c(8), end_month == i)),
      Living_Txp_count     = nrow(filter(dt, CAN_REM_CD %in% c(15), end_month == i)),
      Person_month         = sum(tmp$person_month, na.rm = TRUE)
    ) %>%
    mutate(
      Txp_rate            = Txp_count / Person_month,
      Deceased_Txp_rate   = Deceased_Txp_count / Person_month,
      Txp_free_death_rate = Txp_free_death_count / Person_month,
      Living_Txp_rate     = Living_Txp_count / Person_month
    )
  }) %>% bind_rows()

  write.csv(out, out_csv, row.names = FALSE)
  invisible(out)
}

# ============================================================
# @concept: Post-Transplant Survival
# @fn: compute_post_tx_survival_flags
# @data_kind: transplant
# @needs: data/transplant.csv
# 说明：
# - 生成三类事件指示：compdth（全因移植失败/死亡）、ptx_death（患者死亡）、gft（死亡-删失的移植失败）
# - 需要字段：TFL_DEATH_DT, TFL_GRAFT_DT, PERS_SSA_DEATH_DT, PERS_OPTN_DEATH_DT
# ============================================================
compute_post_tx_survival_flags <- function(
  transplant_csv,
  out_csv = "out_post_tx_survival_flags.csv"
) {
  tx_kdpi <- read_csv(transplant_csv, show_col_types = FALSE)
  tx_kdpi <- tx_kdpi %>%
    mutate(
      compdth   = ifelse(!is.na(TFL_DEATH_DT) | !is.na(TFL_GRAFT_DT) |
                           !is.na(PERS_SSA_DEATH_DT) | !is.na(PERS_OPTN_DEATH_DT), 1, 0),
      ptx_death = ifelse(!is.na(TFL_DEATH_DT) | !is.na(PERS_SSA_DEATH_DT) | !is.na(PERS_OPTN_DEATH_DT), 1, 0),
      gft       = ifelse(!is.na(TFL_GRAFT_DT), 1, 0)
    )

  write.csv(tx_kdpi, out_csv, row.names = FALSE)
  invisible(tx_kdpi)
}

# ============================================================
# @concept: Pre-transplant Mortality Rate
# @fn: compute_pre_tx_mortality_rate
# @data_kind: candidate
# @needs: data/candidate.csv
# 说明：
# - 参照候补名单人月法，计算“等待期间死亡率”（CAN_REM_CD==8）随月份的变化
# - 需要字段：WL_ORG, CAN_REM_DT,CAN_REM_DT,CAN_LISTING_DT,CAN_REM_CD
# ============================================================
compute_pre_tx_mortality_rate <- function(
  candidate_csv,
  out_csv = "out_pre_tx_mortality_rate.csv"
) {
  cand_kipa <- read_csv(candidate_csv, show_col_types = FALSE)

  dt <- cand_kipa %>%
    filter(WL_ORG == 'KI',
           is.na(CAN_REM_DT) | as.numeric(substr(CAN_REM_DT,1,4)) >= 2010,
           !is.na(CAN_LISTING_DT)) %>%
    mutate(
      start_month = pmax(as.numeric(substr(CAN_LISTING_DT,6,7)) + 12*(as.numeric(substr(CAN_LISTING_DT,1,4)) - 2010), 0),
      end_month   = as.numeric(substr(CAN_REM_DT,6,7)) + 12*(as.numeric(substr(CAN_REM_DT,1,4)) - 2010),
      day_start_month = as.numeric(substr(CAN_LISTING_DT,9,10)),
      day_end_month   = as.numeric(substr(CAN_REM_DT,9,10))
    )

  max_m <- suppressWarnings(max(dt$end_month, na.rm = TRUE))
  if (!is.finite(max_m)) max_m <- 1

  calc_days <- function(i) {
    if (i %% 12 %in% c(1,3,5,7,8,10,0)) return(31)
    if (i %% 12 == 2 && (i %/% 12) %% 4 == 0) return(29)
    if (i %% 12 == 2) return(28)
    30
  }
  days_vec <- vapply(1:max_m, calc_days, numeric(1))

  out <- lapply(1:(max_m-1), function(i){
    days_at_month <- days_vec[i]
    tmp <- dt %>%
      filter(start_month <= i, is.na(end_month) | end_month >= i) %>%
      mutate(person_month = case_when(
        is.na(end_month)                ~ 1,
        start_month == end_month       ~ (day_end_month - day_start_month)/days_at_month,
        start_month == i               ~ (days_at_month + 1 - day_start_month)/days_at_month,
        end_month == i                 ~ day_end_month/days_at_month,
        TRUE                           ~ 1
      ))

    tibble(
      month = i,
      death_count   = nrow(filter(dt, CAN_REM_CD == 8, end_month == i)),
      person_months = sum(tmp$person_month, na.rm = TRUE)
    ) %>%
    mutate(pre_tx_mortality_rate = death_count / person_months)
  }) %>% bind_rows()

  write.csv(out, out_csv, row.names = FALSE)
  invisible(out)
}

# ============================================================
# @concept: Post-Transplant Mortality Rate
# @fn: compute_post_tx_mortality_rate
# @data_kind: transplant
# @needs: data/transplant.csv
# 说明：
# - 以移植后死亡事件（TFL_DEATH_DT / PERS_*_DEATH_DT）为分子，按死亡发生的年月聚合/或总体率
# - 这是快速指标示例；后续可扩展为 Kaplan-Meier / Cox（此处只做率）
# - 需要字段：TFL_DEATH_DT, PERS_SSA_DEATH_DT, PERS_OPTN_DEATH_DT
# ============================================================
compute_post_tx_mortality_rate <- function(
  transplant_csv,
  out_csv = "out_post_tx_mortality_rate.csv"
) {
  tx <- read_csv(transplant_csv, show_col_types = FALSE) %>%
    mutate(
      death_dt = coalesce(as_date(TFL_DEATH_DT), as_date(PERS_SSA_DEATH_DT), as_date(PERS_OPTN_DEATH_DT)),
      death_ym = if_else(!is.na(death_dt), format(death_dt, "%Y-%m"), NA_character_)
    )

  # 简单的“每月死亡数”；若需要人月分母，可与随访表/在院时长做更精细的构造
  by_month <- tx %>%
    filter(!is.na(death_ym)) %>%
    group_by(death_ym) %>%
    summarise(death_count = n(), .groups = "drop")

  write.csv(by_month, out_csv, row.names = FALSE)
  invisible(by_month)
}
