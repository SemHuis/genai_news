# =============================================================================
# analysis.R
# =============================================================================
# Longitudinal stylometric analysis of Dutch newspaper articles.
# Implements interrupted time-series (ITS) regression with mixed-effects
# models to detect AI-related stylistic change post-November 2022.
#
# Workflow:
#   1. Load and merge CSV results from article_metrics.py
#   2. Preprocess: parse dates, create ITS variables, filter, scale
#   3. Quality checks: distributions, outliers, article length confound
#   4. Publication-level ITS (OLS + autocorrelation correction)
#   5. Journalist-level mixed-effects ITS (lme4)
#   6. Focal word ITS (negative binomial / OLS)
#   7. Outlet-type comparison (national vs regional vs local)
#   8. Topic-stratified analysis
#   9. Pangram triangulation (mean AI-assistance score + flagged-article audit)
#  10. Composite fingerprint score
#  11. Timeline plots: monthly & quarterly, by outlet type / by outlet / by
#      topic-within-outlet, full date range even where a metric is all-zero
#  12. Main
#
# Dependencies:
#   install.packages(c("tidyverse", "lme4", "lmerTest", "nlme",
#                      "sandwich", "lmtest", "MASS", "patchwork",
#                      "scales", "broom", "broom.mixed", "jsonlite"))
# =============================================================================

# NOTE: MASS is loaded BEFORE tidyverse/dplyr on purpose. MASS::select()
# (used for ridge-regression model selection) and dplyr::select() (used
# throughout this script for column selection) share a name; loading
# MASS first lets dplyr::select() win the masking and behave as expected.
library(MASS)          # glm.nb() for negative binomial
library(tidyverse)
library(lme4)
library(lmerTest)      # p-values for lmer
library(nlme)          # gls() for autocorrelation-corrected OLS
library(sandwich)      # Newey-West standard errors
library(lmtest)        # coeftest()
library(patchwork)     # combine ggplots
library(scales)        # axis formatting
library(broom)         # tidy() for OLS models
library(broom.mixed)   # tidy() for lmer models
library(jsonlite)      # reading Pangram .jsonl / .json results

# =============================================================================
# 0.  CONFIGURATION 
# =============================================================================

# --- Input files ---
# One CSV per newspaper, output of article_metrics.py.
# Outlet codes match article_metrics.py's output naming convention:
# {code}_metrics_results_body.csv
INPUT_FILES <- list(
  vk   = "vk_metrics_results_body.csv",    # Volkskrant
  tgf  = "tgf_metrics_results_body.csv",   # Telegraaf
  dvhn = "dvhn_metrics_results_body.csv",  # Dagblad van het Noorden
  ed   = "ed_metrics_results_body.csv",    # Eindhovens Dagblad
  stc  = "stc_metrics_results_body.csv",   # Steenwijker Courant
  nof  = "nof_metrics_results_body.csv"    # Noordoost Friesland
)

# Outlet type assignment (national / regional / local)
OUTLET_TYPES <- c(
  vk   = "national",
  tgf  = "national",
  dvhn = "regional",
  ed   = "regional",
  stc  = "local",
  nof  = "local"
)

# Full display names for all outlets — used in plot titles, legends and axes.
# Short codes (vk, tgf, …) are kept for file paths and data joins.
OUTLET_NAMES <- c(
  vk   = "De Volkskrant",
  tgf  = "De Telegraaf",
  dvhn = "Dagblad van het Noorden",
  ed   = "Eindhovens Dagblad",
  stc  = "Steenwijker Courant",
  nof  = "Nieuwsblad Noordoost-Friesland"
)

# Outlet-type display order: national → regional → local (used as factor levels)
OUTLET_TYPE_LEVELS <- c("national", "regional", "local")


# Per outlet: pangram_results_{code}.jsonl (full results, every article) and
# ai_flagged_articles_{code}.json (full article records, AI-flagged subset
# only), both under PANGRAM_DIR. Either or both files may be absent for a
# given outlet; missing files are skipped with a message, not an error.
PANGRAM_DIR <- "pangram_results"
PANGRAM_FILES <- setNames(
  file.path(PANGRAM_DIR, paste0("pangram_results_", names(INPUT_FILES), ".jsonl")),
  names(INPUT_FILES)
)
FLAGGED_FILES <- setNames(
  file.path(PANGRAM_DIR, paste0("ai_flagged_articles_", names(INPUT_FILES), ".json")),
  names(INPUT_FILES)
)

# --- Intervention point ---
INTERVENTION_DATE <- as.Date("2022-12-01")  # ChatGPT public release

# --- Output directory for plots ---
PLOT_DIR <- "plots"
dir.create(PLOT_DIR, showWarnings = FALSE)
# Timeline plots (Section 11) are organised under PLOT_DIR by time unit and
# breakdown dimension, since the full combinatorial set is large:
#   plots/monthly|quarterly/by_outlet_type/...
#   plots/monthly|quarterly/by_outlet/...
#   plots/monthly|quarterly/by_topic/{outlet}/...
TIME_UNITS <- c("month", "quarter")
for (unit_dir in paste0(TIME_UNITS, "ly")) {
  for (sub in c("by_outlet_type", "by_outlet", "by_topic")) {
    dir.create(file.path(PLOT_DIR, unit_dir, sub), recursive = TRUE, showWarnings = FALSE)
  }
}

# --- Minimum article length (words) to include ---
MIN_WORDS <- 0

# --- Minimum articles per journalist per month for journalist-level analysis ---
MIN_ARTICLES_JOURNALIST_MONTH <- 3

# --- Minimum months active for journalist to be included in trajectory analysis ---
MIN_MONTHS_JOURNALIST <- 24

# --- Multiple comparison correction method ---
# "BH" (Benjamini-Hochberg FDR) or "bonferroni"
P_ADJUST_METHOD <- "BH"

# --- Metrics to analyse ---
# FINGERPRINT_METRICS is a fixed set of lexical/syntactic columns
# article_metrics.py always produces (modulo NA when their inputs, e.g.
# Stanza or a frequency source, weren't supplied).
FINGERPRINT_METRICS <- c(
  "mtld",
  "prop_low_freq",
  "prop_high_freq",
  "mean_zipf",
  "freq_pronoun",
  "freq_aux",
  "freq_det",
  "freq_cconj",
  "freq_adv",
  "freq_adp",
  "mean_sent_len",
  "cv_sent_len",
  "mean_dep_depth",
  "finite_verbs_per_sent",
  "mean_tunit_len",
  "nom_rate"
)

# FOCAL_METRICS_BASE is a fallback default (top-20 AI-seed metrics) used
# only before main() has run finalize_metric_config() against real data —
# e.g. for interactive use of individual functions. The REAL set of focal
# metrics is discovered at runtime by detect_focal_metrics() / 
# detect_category_metrics() below, since article_metrics.py's --n_seeds can
# produce any combination of top-N AI-seed/baseline columns (default
# 20/50/200) and the category set is configurable via --categories.
#
# article_metrics.py (Juzek 2026 Sec. 3.4-3.6 methodology) produces, per N
# in --n_seeds:
#   llm_style_word_ratio_top{N}        AI-seed (top-N by LPR) rate
#   weighted_llm_ratio_top{N}_per1k    AI-seed rate, weighted by LPR
#   baseline_word_ratio_top{N}         matched near-zero-LPR control rate
#   baseline_weighted_ratio_top{N}_per1k
# plus llm_style_word_count_top{N} / baseline_word_count_top{N} (raw integer
# counts — NOT treated as analysis metrics here; they exist purely as exact
# inputs to the chi-square split-corpus test in Section 9b, which reads them
# directly by name rather than through ALL_METRICS).
FOCAL_METRICS_BASE <- c(
  "llm_style_word_ratio_top20",
  "weighted_llm_ratio_top20_per1k"
)

# Regexes used to recognise the N-suffixed AI-seed / baseline rate columns
# dynamically, since the specific N's present depend on how
# article_metrics.py was run (--n_seeds).
.AI_SEED_RATIO_RE   <- "^llm_style_word_ratio_top[0-9]+$"
.AI_SEED_WEIGHTED_RE <- "^weighted_llm_ratio_top[0-9]+_per1k$"
.BASELINE_RATIO_RE   <- "^baseline_word_ratio_top[0-9]+$"
.BASELINE_WEIGHTED_RE <- "^baseline_weighted_ratio_top[0-9]+_per1k$"
# Raw-count columns are deliberately NOT matched by detect_focal_metrics() —
# they're consumed directly by run_split_corpus_analysis() (Sec. 9b).

detect_focal_metrics <- function(df) {
  # Any top-N AI-seed or matched-baseline rate column actually present in
  # the data (excludes the parallel _count_ columns — see comment above).
  pat <- paste(.AI_SEED_RATIO_RE, .AI_SEED_WEIGHTED_RE,
               .BASELINE_RATIO_RE, .BASELINE_WEIGHTED_RE, sep = "|")
  names(df)[str_detect(names(df), pat)]
}

detect_category_metrics <- function(df) {
  # Any column ending in _freq_per1k that isn't already a known fixed metric
  # is treated as a focal-word semantic-category column (built-in:
  # care_rigour_freq_per1k, emphasize_freq_per1k, importance_freq_per1k —
  # Juzek 2026 Sec. 3.6 — or whatever --categories CSV was used).
  cands <- names(df)[str_detect(names(df), "_freq_per1k$")]
  cands[!cands %in% c(FINGERPRINT_METRICS, FOCAL_METRICS_BASE)]
}

# AI fingerprint directional hypotheses:
# positive = expect INCREASE post-intervention (consistent with more AI)
# negative = expect DECREASE post-intervention
# zero     = no directional hypothesis (excluded from the composite score)
#
# The three built-in semantic categories (Sec. 3.6) and FINGERPRINT_METRICS
# are listed explicitly below. The top-N AI-seed / baseline columns are NOT
# listed by exact name here (since N varies) — get_directional_hypotheses()
# pattern-matches them instead: AI-seed columns default to +1 (more
# AI-overused-word usage = more AI-like, the entire premise of the LPR
# ranking), baseline columns default to 0 (they are Juzek 2026's matched
# *control* condition — near-zero LPR by construction — so no directional
# shift is hypothesised; they're tracked for contrast, not included in the
# composite). Any other, genuinely unrecognised metric still defaults to +1
# with a printed note, as before.
DIRECTIONAL_HYPOTHESES_BASE <- c(
  mtld                      = -1,  # lower lexical diversity
  prop_low_freq             = -1,  # fewer rare words
  prop_high_freq            =  1,  # more common words
  mean_zipf                 =  1,  # higher mean frequency (more common vocab)
  freq_pronoun              =  1,  # more pronouns
  freq_aux                  =  1,  # more auxiliaries
  freq_det                  =  1,  # more determiners
  freq_cconj                =  1,  # more coordinating conjunctions
  freq_adv                  = -1,  # fewer adverbs
  freq_adp                  =  0,  # no strong hypothesis
  mean_sent_len             = -1,  # shorter sentences (syntactic flattening)
  cv_sent_len               = -1,  # lower sentence length variance (less burstiness)
  mean_dep_depth            = -1,  # shallower dependency trees
  finite_verbs_per_sent     = -1,  # fewer clauses (simpler structure)
  mean_tunit_len            = -1,  # shorter T-units
  nom_rate                  =  1,  # more nominalisations
  care_rigour_freq_per1k    =  1,
  emphasize_freq_per1k      =  1,
  importance_freq_per1k     =  1
)

get_directional_hypotheses <- function(metrics) {
  out <- setNames(rep(NA_real_, length(metrics)), metrics)

  known <- intersect(metrics, names(DIRECTIONAL_HYPOTHESES_BASE))
  out[known] <- DIRECTIONAL_HYPOTHESES_BASE[known]

  # Pattern-based defaults for top-N AI-seed / baseline columns, before
  # falling through to the generic "unknown -> +1" default below.
  still_na <- metrics[is.na(out)]
  ai_seed  <- still_na[str_detect(still_na, paste(.AI_SEED_RATIO_RE, .AI_SEED_WEIGHTED_RE, sep = "|"))]
  baseline <- still_na[str_detect(still_na, paste(.BASELINE_RATIO_RE, .BASELINE_WEIGHTED_RE, sep = "|"))]
  out[ai_seed]  <- 1
  out[baseline] <- 0

  unknown <- metrics[is.na(out)]
  if (length(unknown) > 0) {
    message("[hypothesis] No directional hypothesis set for: ",
            paste(unknown, collapse = ", "),
            " — defaulting to +1 (more = more AI-like).")
    out[unknown] <- 1
  }
  out
}

# FOCAL_METRICS and ALL_METRICS depend on which top-N / category columns are
# actually present in the loaded data, so they're finalised inside main()
# (via finalize_metric_config()) rather than as static constants here.
# DIRECTIONAL_HYPOTHESES is likewise finalised there. They still exist as
# global bindings (assigned below) purely so every function's
# `metrics = ALL_METRICS` / `DIRECTIONAL_HYPOTHESES[...]` default argument
# has *something* valid to point to before main() runs — e.g. for
# interactive use of individual functions without calling main() first.
FOCAL_METRICS <- FOCAL_METRICS_BASE
ALL_METRICS <- c(FINGERPRINT_METRICS, FOCAL_METRICS)
DIRECTIONAL_HYPOTHESES <- get_directional_hypotheses(ALL_METRICS)

finalize_metric_config <- function(df) {
  # Re-derives FOCAL_METRICS / ALL_METRICS / DIRECTIONAL_HYPOTHESES from the
  # top-N AI-seed/baseline and category columns actually present in `df`,
  # and pushes them to the global environment (<<-) so every function's
  # default `metrics = ALL_METRICS` argument picks up the data-driven set
  # for the rest of the run. Called once near the top of main(), right
  # after preprocess().
  focal_n_metrics  <- detect_focal_metrics(df)
  category_metrics <- detect_category_metrics(df)
  if (length(focal_n_metrics) > 0) {
    message("[metrics] Top-N AI-seed/baseline columns detected in data: ",
            paste(focal_n_metrics, collapse = ", "))
  }
  if (length(category_metrics) > 0) {
    message("[metrics] Focal categories detected in data: ",
            paste(category_metrics, collapse = ", "))
  }
  focal <- unique(c(focal_n_metrics, category_metrics))
  if (length(focal) == 0) {
    message("[metrics] No focal-word columns detected — falling back to ",
            "static defaults: ", paste(FOCAL_METRICS_BASE, collapse = ", "))
    focal <- FOCAL_METRICS_BASE
  }
  all_m <- c(FINGERPRINT_METRICS, focal)

  FOCAL_METRICS <<- focal
  ALL_METRICS <<- all_m
  DIRECTIONAL_HYPOTHESES <<- get_directional_hypotheses(all_m)
  invisible(all_m)
}

# =============================================================================
# 1.  DATA LOADING AND MERGING
# =============================================================================

load_data <- function(input_files, outlet_types) {
  dfs <- list()
  for (outlet in names(input_files)) {
    path <- input_files[[outlet]]
    if (!file.exists(path)) {
      message("[load] File not found, skipping: ", path)
      next
    }
    df <- read_csv(path, show_col_types = FALSE)
    df$outlet      <- outlet
    df$outlet_type <- outlet_types[[outlet]]
    dfs[[outlet]]  <- df
  }
  if (length(dfs) == 0) stop("No input files found. Check INPUT_FILES paths.")
  bind_rows(dfs)
}

# =============================================================================
# 1b.  PANGRAM RESULTS — LOADING AND MERGING
# =============================================================================
# Two distinct Pangram inputs per outlet, both optional:
#   pangram_results_{code}.jsonl    full results, every article
#   ai_flagged_articles_{code}.json full article records, AI-flagged subset
# Neither carries our own stylometric/focal metrics — both are joined back
# onto the metrics data frame by (title, date) within each outlet.
#
# Known limitation: (title, date) is not a guaranteed-unique key. In the
# dvhn sample, 11 of 6076 (title, date) pairs collide, all generic
# placeholder titles ("No Headline In Original") repeated on the same day.
# A left_join on a colliding key fans out to multiple matching rows; this
# is a small, documented limitation rather than a bug, since article_index
# isn't shared between the Pangram files and article_metrics.py's CSVs.
# =============================================================================

load_pangram_jsonl <- function(path) {
  raw <- jsonlite::stream_in(file(path), verbose = FALSE)
  raw$pangram_ai_score <- sapply(raw$windows, function(w) {
    if (is.null(w) || !is.data.frame(w) || nrow(w) == 0) return(NA_real_)
    mean(w$ai_assistance_score, na.rm = TRUE)
  })
  raw %>%
    transmute(
      title    = title,
      date     = dmy(date),
      pangram_ai_score              = pangram_ai_score,
      pangram_fraction_ai           = fraction_ai,
      pangram_fraction_ai_assisted  = fraction_ai_assisted,
      pangram_fraction_human        = fraction_human,
      pangram_num_ai_segments       = num_ai_segments,
      pangram_had_error             = !is.na(error)
    )
}

attach_pangram_results <- function(df, pangram_files = PANGRAM_FILES) {
  message("\n=== ATTACHING PANGRAM RESULTS ===")
  pieces <- list()
  for (outlet in unique(df$outlet)) {
    od <- df %>% filter(outlet == !!outlet)
    path <- pangram_files[[outlet]]
    if (is.null(path) || !file.exists(path)) {
      message("[pangram] No pangram_results file for outlet '", outlet, "', skipping.")
      pieces[[outlet]] <- od
      next
    }
    pg <- load_pangram_jsonl(path)
    n_before <- nrow(od)
    od <- left_join(od, pg, by = c("title", "date"))
    n_matched <- sum(!is.na(od$pangram_ai_score))
    message("[pangram] ", outlet, ": ", n_matched, "/", n_before,
            " articles matched to a Pangram result (",
            round(100 * n_matched / n_before, 1), "%).")
    if (nrow(od) != n_before) {
      message("[pangram] ", outlet, ": NOTE row count changed ", n_before,
              " -> ", nrow(od), " due to duplicate (title, date) keys in the ",
              "Pangram file — see Section 1b comment.")
    }
    pieces[[outlet]] <- od
  }
  bind_rows(pieces)
}

load_flagged_articles <- function(path) {
  raw <- jsonlite::fromJSON(path, simplifyDataFrame = TRUE)
  raw$author_clean <- sapply(raw$author, function(a) paste(a, collapse = "; "))
  raw %>%
    transmute(
      title          = title,
      date           = dmy(date),
      flagged_author = author_clean,
      flagged_section = section,
      flagged_predicted_topic = predicted_topic,
      flagged_fraction_ai = fraction_ai,
      flagged_fraction_ai_assisted = fraction_ai_assisted
    )
}

report_flagged_article_metrics <- function(df, flagged_files = FLAGGED_FILES,
                                            metrics = ALL_METRICS) {
  message("\n=== AI-FLAGGED ARTICLE METRICS AUDIT ===")
  metrics <- metrics[metrics %in% names(df)]
  pangram_cols <- intersect(
    c("pangram_ai_score", "pangram_fraction_ai", "pangram_fraction_ai_assisted",
      "pangram_fraction_human", "pangram_num_ai_segments"),
    names(df)
  )
  id_cols <- intersect(c("outlet", "outlet_type", "title", "date", "author_clean",
                         "predicted_topic", "total_words"), names(df))

  pieces <- list()
  for (outlet in names(flagged_files)) {
    path <- flagged_files[[outlet]]
    if (!file.exists(path)) {
      message("[flagged] No ai_flagged_articles file for outlet '", outlet, "', skipping.")
      next
    }
    flagged <- load_flagged_articles(path)
    od <- df %>% filter(outlet == !!outlet)

    matched <- inner_join(flagged, od, by = c("title", "date"))
    message("[flagged] ", outlet, ": ", n_distinct(matched$title), "/",
            n_distinct(flagged$title), " flagged articles matched in metrics data.")

    if (nrow(matched) > 0) {
      pieces[[outlet]] <- matched %>%
        dplyr::select(all_of(c(id_cols, pangram_cols, metrics,
                               "flagged_fraction_ai", "flagged_fraction_ai_assisted")))
    }
  }

  if (length(pieces) == 0) {
    message("[flagged] No flagged articles matched any outlet's metrics data.")
    return(invisible(NULL))
  }

  out <- bind_rows(pieces)
  fname <- file.path(PLOT_DIR, "flagged_articles_metrics.csv")
  write_csv(out, fname)
  message("[flagged] Wrote ", nrow(out), " flagged-article metric rows to: ", fname)
  invisible(out)
}

# =============================================================================
# 2.  PREPROCESSING
# =============================================================================

preprocess <- function(df, min_words = MIN_WORDS,
                       intervention_date = INTERVENTION_DATE) {
  # Parse date
  df <- df %>%
    mutate(
      date = dmy(date),                            # day/month/year format
      year_month = floor_date(date, "month"),
      month_num  = as.integer(
        difftime(year_month,
                 min(year_month, na.rm = TRUE),
                 units = "days") / 30.44
      )
    ) %>%
    filter(!is.na(date))
  
  # ITS variables
  df <- df %>%
    mutate(
      # Time (continuous month index from start of corpus)
      time = month_num,
      
      # Intervention dummy: 0 before, 1 from November 2022 onwards
      intervention = as.integer(date >= intervention_date),
      
      # Post-intervention time counter (0 before, 1, 2, 3... after)
      time_after = pmax(0, as.integer(
        difftime(floor_date(date, "month"),
                 floor_date(intervention_date, "month"),
                 units = "days") / 30.44
      ) * as.integer(date >= intervention_date))
    )
  
  # Period label for summaries and plots
  df <- df %>%
    mutate(period = ifelse(intervention == 0, "Pre-ChatGPT", "Post-ChatGPT"))
  
  # Remove very short articles
  df <- df %>% filter(!is.na(total_words), total_words >= min_words)
  
  # Clean author column: take first author if multiple
  df <- df %>%
    mutate(
      author_clean = str_trim(str_split_fixed(author, ";", 2)[, 1]),
      author_clean = na_if(author_clean, "")
    )
  
  # Journalist ID: outlet + author name (to distinguish same name across outlets)
  df <- df %>%
    mutate(journalist_id = paste(outlet, author_clean, sep = "::"))

  # Full outlet names for display in plots; ordered outlet_type factor so
  # legends always read national → regional → local.
  df <- df %>%
    mutate(
      outlet_name  = unname(OUTLET_NAMES[outlet]),
      outlet_type  = factor(outlet_type, levels = OUTLET_TYPE_LEVELS)
    )

  message("[preprocess] Rows after filtering: ", nrow(df))
  message("[preprocess] Date range: ",
          min(df$date, na.rm = TRUE), " to ",
          max(df$date, na.rm = TRUE))
  message("[preprocess] Outlets: ", paste(unique(df$outlet), collapse = ", "))
  
  df
}

# =============================================================================
# 3.  QUALITY CHECKS
# =============================================================================

quality_checks <- function(df, metrics = ALL_METRICS) {
  message("\n=== QUALITY CHECKS ===")
  
  # 3.1 Metric coverage
  message("\n[check] Metric coverage (% non-NA):")
  coverage <- sapply(metrics, function(m) {
    if (!m %in% names(df)) return(NA_real_)
    mean(!is.na(df[[m]])) * 100
  })
  print(round(coverage, 1))
  
  # 3.2 Article length distribution
  message("\n[check] Article length (total_words):")
  print(summary(df$total_words))
  
  # 3.3 Check for article length ~ time trend (potential confound)
  message("\n[check] Article length trend over time (should be flat):")
  len_trend <- lm(total_words ~ time, data = df)
  print(tidy(len_trend))
  
  # 3.4 Outlet × period balance
  message("\n[check] Articles per outlet per period:")
  print(df %>%
          count(outlet, period) %>%
          pivot_wider(names_from = period, values_from = n))
  
  # 3.5 Topic distribution stability
  if ("predicted_topic" %in% names(df)) {
    message("\n[check] Topic distribution pre vs post (top topics):")
    print(df %>%
            count(outlet, period, predicted_topic) %>%
            group_by(outlet, period) %>%
            mutate(pct = n / sum(n) * 100) %>%
            arrange(outlet, period, desc(pct)) %>%
            slice_head(n = 5))
  }
  
  invisible(df)
}

# =============================================================================
# 4.  PUBLICATION-LEVEL ITS
# =============================================================================
# Aggregates to outlet-month means, then fits ITS regression.
# Uses Newey-West standard errors to correct for autocorrelation.
# =============================================================================

aggregate_outlet_month <- function(df, metrics = ALL_METRICS) {
  covariates <- c("outlet", "outlet_type", "year_month", "time",
                  "intervention", "time_after")
  
  if ("predicted_topic" %in% names(df)) {
    # Topic composition covariate: entropy of topic distribution per outlet-month
    topic_entropy <- df %>%
      filter(!is.na(predicted_topic)) %>%
      count(outlet, year_month, predicted_topic) %>%
      group_by(outlet, year_month) %>%
      mutate(p = n / sum(n)) %>%
      summarise(topic_entropy = -sum(p * log(p + 1e-9)), .groups = "drop")
  }
  
  agg_metrics <- df %>%
    group_by(across(all_of(covariates))) %>%
    summarise(
      n_articles   = n(),
      mean_words   = mean(total_words, na.rm = TRUE),
      across(all_of(metrics[metrics %in% names(df)]),
             ~ mean(.x, na.rm = TRUE)),
      .groups = "drop"
    )
  
  if ("predicted_topic" %in% names(df)) {
    agg_metrics <- left_join(agg_metrics, topic_entropy,
                             by = c("outlet", "year_month"))
  }
  
  agg_metrics
}


run_outlet_its <- function(agg_df, metric,
                           use_newey_west = TRUE,
                           lag_max = 3) {
  # Filter to complete cases for this metric
  d <- agg_df %>%
    filter(!is.na(.data[[metric]])) %>%
    arrange(outlet, year_month)
  
  results <- list()
  
  for (o in unique(d$outlet)) {
    od <- d %>% filter(outlet == o)
    if (nrow(od) < 20) next
    
    formula_str <- paste(metric,
                         "~ time + intervention + time_after + mean_words")
    if ("topic_entropy" %in% names(od)) {
      formula_str <- paste(formula_str, "+ topic_entropy")
    }
    
    fit <- lm(as.formula(formula_str), data = od)
    
    if (use_newey_west) {
      # Newey-West HAC standard errors — robust to autocorrelation
      coef_nw <- coeftest(fit, vcov = NeweyWest(fit, lag = lag_max,
                                                prewhite = FALSE))
      # NOTE: coef_nw has class "coeftest" (no "matrix" in its class vector),
      # so as.data.frame() does not dispatch to as.data.frame.matrix and
      # instead wraps the whole matrix into a single malformed column.
      # unclass() restores plain-matrix dispatch.
      tidy_res <- as.data.frame(unclass(coef_nw)) %>%
        rownames_to_column("term") %>%
        rename(estimate = Estimate,
               std.error = `Std. Error`,
               statistic = `t value`,
               p.value   = `Pr(>|t|)`)
    } else {
      tidy_res <- tidy(fit)
    }
    
    tidy_res$outlet <- o
    tidy_res$metric <- metric
    tidy_res$n_obs  <- nrow(od)
    tidy_res$r_squared <- summary(fit)$r.squared
    results[[o]] <- tidy_res
  }
  
  bind_rows(results)
}


run_all_outlet_its <- function(agg_df,
                               metrics = ALL_METRICS,
                               p_adjust = P_ADJUST_METHOD) {
  message("\n=== PUBLICATION-LEVEL ITS ===")
  all_results <- list()
  for (m in metrics) {
    if (!m %in% names(agg_df)) next
    message("  [its] ", m)
    res <- run_outlet_its(agg_df, m)
    all_results[[m]] <- res
  }
  
  results_df <- bind_rows(all_results)
  
  # Extract key coefficients and apply FDR correction within families
  key_coefs <- results_df %>%
    filter(term %in% c("intervention", "time_after", "time")) %>%
    group_by(term) %>%
    mutate(p.adjusted = p.adjust(p.value, method = p_adjust)) %>%
    ungroup()
  
  # Add directional hypothesis check
  key_coefs <- key_coefs %>%
    mutate(
      expected_direction = DIRECTIONAL_HYPOTHESES[metric],
      direction_matches  = case_when(
        expected_direction ==  1 ~ estimate > 0,
        expected_direction == -1 ~ estimate < 0,
        TRUE ~ NA
      ),
      significant        = p.adjusted < 0.05,
      significant_correct_direction = significant & (direction_matches %in% TRUE)
    )
  
  message("\n[its] Significant level shifts (intervention term, correct direction):")
  print(key_coefs %>%
          filter(term == "intervention", significant_correct_direction) %>%
          dplyr::select(metric, outlet, estimate, std.error, p.value, p.adjusted) %>%
          arrange(p.adjusted))
  
  list(full = results_df, key = key_coefs)
}

# =============================================================================
# 5.  JOURNALIST-LEVEL MIXED-EFFECTS ITS
# =============================================================================
# lmer: metric ~ time + intervention + time_after + total_words +
#               outlet_type + [topic] + (1 + time | journalist_id) + (1 | outlet)
# Random intercept + slope per journalist: captures individual trajectories
# Random intercept per outlet: accounts for nested structure
# =============================================================================

filter_journalists <- function(df,
                               min_months   = MIN_MONTHS_JOURNALIST,
                               min_per_month = MIN_ARTICLES_JOURNALIST_MONTH) {
  # Keep journalists with sufficient activity across the time window
  journalist_stats <- df %>%
    filter(!is.na(journalist_id), journalist_id != "::") %>%
    group_by(journalist_id, year_month) %>%
    summarise(n_month = n(), .groups = "drop") %>%
    group_by(journalist_id) %>%
    summarise(
      n_months       = n_distinct(year_month),
      mean_per_month = mean(n_month),
      .groups = "drop"
    ) %>%
    filter(n_months >= min_months, mean_per_month >= min_per_month)
  
  message("[journalists] Journalists meeting criteria: ", nrow(journalist_stats))
  df %>% filter(journalist_id %in% journalist_stats$journalist_id)
}


run_journalist_its <- function(df_journalists, metric,
                               include_topic = TRUE) {
  d <- df_journalists %>%
    filter(!is.na(.data[[metric]]), !is.na(journalist_id))
  
  if (nrow(d) < 200) {
    message("[lmer] Too few observations for ", metric, ", skipping.")
    return(NULL)
  }
  
  # Build formula
  fixed  <- paste(metric, "~ time + intervention + time_after + total_words")
  # outlet_type is only estimable as a fixed effect with >1 level present
  if (n_distinct(d$outlet_type) > 1) {
    fixed <- paste(fixed, "+ outlet_type")
  }
  if (include_topic && "predicted_topic" %in% names(d)) {
    fixed <- paste(fixed, "+ predicted_topic")
  }
  # (1 | outlet) is only identifiable with >1 outlet level; drop it
  # otherwise rather than letting every metric's lmer() fit fail.
  random <- if (n_distinct(d$outlet) > 1) {
    "(1 + time | journalist_id) + (1 | outlet)"
  } else {
    "(1 + time | journalist_id)"
  }
  formula_str <- paste(fixed, "+", random)
  
  tryCatch({
    fit <- lmer(as.formula(formula_str), data = d,
                control = lmerControl(optimizer = "bobyqa",
                                      optCtrl   = list(maxfun = 2e5)))
    list(
      model  = fit,
      tidy   = tidy(fit, effects = "fixed", conf.int = TRUE),
      random = tidy(fit, effects = "ran_vals"),
      metric = metric
    )
  }, error = function(e) {
    message("[lmer] Error for ", metric, ": ", e$message)
    NULL
  })
}


run_all_journalist_its <- function(df_journalists,
                                   metrics = ALL_METRICS,
                                   p_adjust = P_ADJUST_METHOD) {
  message("\n=== JOURNALIST-LEVEL MIXED-EFFECTS ITS ===")
  results <- list()
  for (m in metrics) {
    if (!m %in% names(df_journalists)) next
    message("  [lmer] ", m)
    res <- run_journalist_its(df_journalists, m)
    if (!is.null(res)) results[[m]] <- res
  }
  
  # Extract fixed effects and apply FDR correction
  if (length(results) == 0) {
    message("[lmer] No successful models for any metric; skipping fixed-effect summary.")
    return(list(models = results,
                fixed  = tibble(metric = character(), term = character(),
                                estimate = double(), std.error = double(),
                                p.value = double())))
  }

  fixed_df <- map_dfr(results, ~ .x$tidy, .id = "metric") %>%
    filter(term %in% c("intervention", "time_after", "time")) %>%
    group_by(term) %>%
    mutate(p.adjusted = p.adjust(p.value, method = p_adjust)) %>%
    ungroup() %>%
    mutate(
      expected_direction = DIRECTIONAL_HYPOTHESES[metric],
      direction_matches  = case_when(
        expected_direction ==  1 ~ estimate > 0,
        expected_direction == -1 ~ estimate < 0,
        TRUE ~ NA
      ),
      significant = p.adjusted < 0.05,
      significant_correct_direction = significant & (direction_matches %in% TRUE)
    )
  
  message("\n[lmer] Significant intervention effects (correct direction):")
  print(fixed_df %>%
          filter(term == "intervention", significant_correct_direction) %>%
          dplyr::select(metric, estimate, std.error, p.value, p.adjusted) %>%
          arrange(p.adjusted))
  
  list(models = results, fixed = fixed_df)
}

# =============================================================================
# 6.  FOCAL WORD NEGATIVE BINOMIAL MODELS
# =============================================================================
# Focal word counts are integers; negative binomial handles overdispersion.
# Response: raw count; offset: log(total_words) to model rate.
# =============================================================================

run_focal_negbin <- function(agg_df, count_col, offset_col = "n_articles",
                             p_adjust = P_ADJUST_METHOD) {
  message("\n=== FOCAL WORD NEGATIVE BINOMIAL ITS ===")
  # Use outlet-month aggregated counts if available
  # Otherwise fall back to article-level OLS on ratio
  
  results <- list()
  for (o in unique(agg_df$outlet)) {
    od <- agg_df %>% filter(outlet == o, !is.na(.data[[count_col]]))
    if (nrow(od) < 20) next
    
    tryCatch({
      # Approximate count from ratio × n_articles for negative binomial
      # If you have raw counts from Python, use those directly instead
      od$approx_count <- round(od[[count_col]] * od[[offset_col]])
      
      fit <- glm.nb(
        approx_count ~ time + intervention + time_after +
          offset(log(n_articles + 1)),
        data = od
      )
      res <- tidy(fit) %>%
        mutate(outlet = o, metric = count_col)
      results[[o]] <- res
    }, error = function(e) {
      message("  [negbin] Error for ", o, ": ", e$message)
    })
  }
  
  bind_rows(results)
}

# =============================================================================
# 7.  OUTLET-TYPE COMPARISON
# =============================================================================
# Adds outlet_type × intervention interaction to test whether national/
# regional/local outlets show different post-2022 shifts.
# =============================================================================

run_outlet_type_comparison <- function(df_journalists, metric) {
  d <- df_journalists %>%
    filter(!is.na(.data[[metric]]), !is.na(outlet_type)) %>%
    mutate(outlet_type = factor(outlet_type,
                                levels = c("national", "regional", "local")))
  
  if (nrow(d) < 200) return(NULL)
  
  formula_str <- paste(
    metric,
    "~ time + intervention * outlet_type + time_after * outlet_type",
    "+ total_words + (1 | journalist_id) + (1 | outlet)"
  )
  
  tryCatch({
    fit <- lmer(as.formula(formula_str), data = d,
                control = lmerControl(optimizer = "bobyqa"))
    tidy(fit, effects = "fixed", conf.int = TRUE) %>%
      mutate(metric = metric)
  }, error = function(e) {
    message("[outlet_type] Error for ", metric, ": ", e$message)
    NULL
  })
}


run_all_outlet_type_comparisons <- function(df_journalists,
                                            metrics = ALL_METRICS) {
  message("\n=== OUTLET-TYPE COMPARISON ===")
  map_dfr(metrics[metrics %in% names(df_journalists)],
          ~ run_outlet_type_comparison(df_journalists, .x))
}

# =============================================================================
# 8.  TOPIC-STRATIFIED ANALYSIS
# =============================================================================

run_topic_stratified <- function(agg_df, metric,
                                 min_obs = 20) {
  if (!"predicted_topic" %in% names(agg_df)) return(NULL)
  
  results <- list()
  for (topic in unique(agg_df$predicted_topic)) {
    td <- agg_df %>%
      filter(predicted_topic == topic, !is.na(.data[[metric]]))
    if (nrow(td) < min_obs) next
    
    formula_str <- paste(metric,
                         "~ time + intervention + time_after + mean_words")
    fit <- lm(as.formula(formula_str), data = td)
    coef_nw <- coeftest(fit, vcov = NeweyWest(fit, lag = 3,
                                              prewhite = FALSE))
    # See note in run_outlet_its(): unclass() avoids a broken
    # as.data.frame() dispatch on "coeftest" objects.
    res <- as.data.frame(unclass(coef_nw)) %>%
      rownames_to_column("term") %>%
      rename(estimate = Estimate, std.error = `Std. Error`,
             statistic = `t value`, p.value = `Pr(>|t|)`) %>%
      mutate(topic = topic, metric = metric)
    results[[topic]] <- res
  }
  
  bind_rows(results)
}

# =============================================================================
# 9.  PANGRAM TRIANGULATION
# =============================================================================

run_pangram_analysis <- function(df) {
  if (!"pangram_ai_score" %in% names(df)) {
    message("[pangram] No pangram_ai_score column found (attach_pangram_results() ",
            "wasn't run, or no outlet had a matching pangram_results file). Skipping.")
    return(NULL)
  }

  message("\n=== PANGRAM TRIANGULATION ===")

  # 9.1 ITS on mean monthly Pangram AI-assistance score per outlet
  # NOTE: run_outlet_its()'s formula always includes `mean_words` (see
  # Section 4) — this aggregate needs that column too, the same way
  # aggregate_outlet_month() provides it for the main metrics.
  agg_pangram <- df %>%
    group_by(outlet, outlet_type, year_month, time, intervention, time_after) %>%
    summarise(
      mean_pangram = mean(pangram_ai_score, na.rm = TRUE),
      mean_words   = mean(total_words, na.rm = TRUE),
      n_articles   = n(),
      .groups = "drop"
    )

  pangram_its <- run_outlet_its(agg_pangram, "mean_pangram")

  # 9.2 Validate stylometric metrics against Pangram scores
  # Which metrics best predict Pangram scores?
  validation_metrics <- FINGERPRINT_METRICS[FINGERPRINT_METRICS %in% names(df)]
  d_val <- df %>% filter(!is.na(pangram_ai_score))

  if (length(validation_metrics) > 0 && nrow(d_val) > 100) {
    val_formula <- paste(
      "pangram_ai_score ~",
      paste(validation_metrics, collapse = " + "),
      "+ total_words"
    )
    # outlet_type is only estimable with >1 level present — degenerate
    # (and lm() errors on contrasts) if Pangram data only exists for
    # outlets of a single type, e.g. when only one outlet has a
    # pangram_results file.
    if (n_distinct(d_val$outlet_type) > 1) {
      val_formula <- paste(val_formula, "+ outlet_type")
    }
    val_fit <- lm(as.formula(val_formula), data = d_val)
    message("[pangram] Stylometric metrics ~ Pangram R²: ",
            round(summary(val_fit)$r.squared, 3))
    message("[pangram] Significant predictors of Pangram score:")
    print(tidy(val_fit) %>%
            filter(p.value < 0.05) %>%
            arrange(p.value) %>%
            dplyr::select(term, estimate, p.value))
  }

  list(its = pangram_its, agg = agg_pangram)
}

# =============================================================================
# 9b. SPLIT-CORPUS PRE/POST ANALYSIS  (Juzek 2026 Sec. 3.5 replication)
# =============================================================================
# A clean, DELIBERATELY SEPARATE binary before/after comparison — distinct
# from the continuous ITS regression in Sections 4-8, which models
# `intervention` and `time_after` as continuous terms within one trend line
# spanning the whole corpus. This section instead splits the corpus into
# exactly two halves at INTERVENTION_DATE and tests, for each top-N
# AI-seed word set and its matched near-zero-LPR baseline (see
# article_metrics.py's load_focal_words()), whether aggregate prevalence
# shifted between the two halves — mirroring Juzek (2026) Sec. 3.5's
# pre-/post-ChatGPT chi-square design as closely as this corpus allows
# (there: 2020-2021 vs 2023-2024 WMT periods, contingency tables of
# top-20 AI lemma+UPOS-key counts against all remaining tokens in each
# period; here: before/after INTERVENTION_DATE within this corpus, run
# per-N and per-outlet plus pooled across the whole corpus).
#
# This intentionally does NOT touch agg / df_journalists / the ITS model
# functions above, and is not folded into generate_timeline_plots() either
# — it is its own self-contained pre/post test, on raw token counts, kept
# out of the continuous-time machinery so the two designs can't be
# conflated when reading results.
# =============================================================================

split_corpus_period <- function(df, intervention_date = INTERVENTION_DATE) {
  df %>%
    mutate(split_period = if_else(date < intervention_date, "pre", "post"))
}

run_split_corpus_chisq <- function(df, n, group_col = "outlet") {
  # One N (e.g. 20/50/200): 2x2 contingency tables of [top-N AI-seed raw
  # token count] vs [all remaining tokens], pre vs post period, and the
  # same construction for the matched baseline set, each tested
  # independently with its own chi-square test (matching Fig. 2's design:
  # AI-set and baseline-set shifts are tested and reported separately, then
  # compared visually/in aggregate — not combined into one 2x4 table).
  # Run once pooled across the whole corpus and once per group_col level
  # (our analogue of the paper's per-language tests).
  ai_count_col <- paste0("llm_style_word_count_top", n)
  bl_count_col <- paste0("baseline_word_count_top", n)
  if (!all(c(ai_count_col, bl_count_col, "total_words") %in% names(df))) {
    message("[split-corpus] top-", n, ": required count columns not found, skipping.")
    return(NULL)
  }

  run_test <- function(count_pre, count_post, total_pre, total_post) {
    tab <- matrix(
      c(count_pre, total_pre - count_pre, count_post, total_post - count_post),
      nrow = 2,
      dimnames = list(c("target_set", "all_other_tokens"), c("pre", "post"))
    )
    if (any(is.na(tab)) || any(tab < 0)) return(NULL)
    tryCatch(chisq.test(tab), error = function(e) NULL)
  }

  run_one <- function(d, label) {
    agg <- d %>%
      filter(!is.na(split_period)) %>%
      group_by(split_period) %>%
      summarise(
        ai_count    = sum(.data[[ai_count_col]], na.rm = TRUE),
        bl_count    = sum(.data[[bl_count_col]], na.rm = TRUE),
        total_count = sum(total_words, na.rm = TRUE),
        n_articles  = n(),
        .groups = "drop"
      )
    if (!all(c("pre", "post") %in% agg$split_period)) return(NULL)

    pre  <- agg %>% filter(split_period == "pre")
    post <- agg %>% filter(split_period == "post")
    if (pre$total_count == 0 || post$total_count == 0) return(NULL)

    ai_test <- run_test(pre$ai_count, post$ai_count, pre$total_count, post$total_count)
    bl_test <- run_test(pre$bl_count, post$bl_count, pre$total_count, post$total_count)

    ai_pre_rate  <- pre$ai_count  / pre$total_count  * 1000
    ai_post_rate <- post$ai_count / post$total_count * 1000
    bl_pre_rate  <- pre$bl_count  / pre$total_count  * 1000
    bl_post_rate <- post$bl_count / post$total_count * 1000

    tibble(
      group   = label,
      n_seeds = n,
      ai_pre_rate_per1k  = ai_pre_rate,
      ai_post_rate_per1k = ai_post_rate,
      ai_pct_change      = if (ai_pre_rate > 0) (ai_post_rate - ai_pre_rate) / ai_pre_rate * 100 else NA_real_,
      ai_chisq    = if (!is.null(ai_test)) unname(ai_test$statistic) else NA_real_,
      ai_p_value  = if (!is.null(ai_test)) ai_test$p.value else NA_real_,
      baseline_pre_rate_per1k  = bl_pre_rate,
      baseline_post_rate_per1k = bl_post_rate,
      baseline_pct_change = if (bl_pre_rate > 0) (bl_post_rate - bl_pre_rate) / bl_pre_rate * 100 else NA_real_,
      baseline_chisq   = if (!is.null(bl_test)) unname(bl_test$statistic) else NA_real_,
      baseline_p_value = if (!is.null(bl_test)) bl_test$p.value else NA_real_,
      n_articles_pre  = pre$n_articles,
      n_articles_post = post$n_articles
    )
  }

  pieces <- list("(pooled)" = run_one(df, "(pooled)"))
  if (!is.null(group_col) && group_col %in% names(df)) {
    for (g in sort(unique(df[[group_col]]))) {
      pieces[[g]] <- run_one(df %>% filter(.data[[group_col]] == g), g)
    }
  }
  bind_rows(pieces)
}

run_split_corpus_analysis <- function(df, n_seeds_used = NULL,
                                      intervention_date = INTERVENTION_DATE,
                                      p_adjust = P_ADJUST_METHOD) {
  message("\n=== SPLIT-CORPUS PRE/POST ANALYSIS (Juzek 2026 Sec. 3.5) ===")
  message("[split-corpus] Deliberately separate binary before/after test at ",
          intervention_date, " — distinct from the continuous ITS regression",
          " in Sections 4-8.")

  if (is.null(n_seeds_used)) {
    count_cols <- names(df)[str_detect(names(df), "^llm_style_word_count_top[0-9]+$")]
    n_seeds_used <- sort(as.integer(str_extract(count_cols, "[0-9]+$")))
  }
  if (length(n_seeds_used) == 0) {
    message("[split-corpus] No llm_style_word_count_top{N} columns found ",
            "— skipping (was article_metrics.py run with --wordlist?).")
    return(NULL)
  }

  d <- split_corpus_period(df, intervention_date)

  all_results <- map_dfr(n_seeds_used, function(n) run_split_corpus_chisq(d, n))
  if (nrow(all_results) == 0) {
    message("[split-corpus] No results produced.")
    return(NULL)
  }

  all_results <- all_results %>%
    group_by(n_seeds) %>%
    mutate(
      ai_p_adjusted       = p.adjust(ai_p_value, method = p_adjust),
      baseline_p_adjusted = p.adjust(baseline_p_value, method = p_adjust)
    ) %>%
    ungroup()

  fname <- file.path(PLOT_DIR, "split_corpus_pre_post.csv")
  write_csv(all_results, fname)
  message("[split-corpus] Wrote results to: ", fname)

  pooled <- all_results %>% filter(group == "(pooled)")
  for (i in seq_len(nrow(pooled))) {
    r <- pooled[i, ]
    message(sprintf(
      "[split-corpus] top-%d (pooled, n=%d pre / %d post articles): AI %+.1f%% (chisq p=%.4g, adj=%.4g) | baseline %+.1f%% (chisq p=%.4g, adj=%.4g)",
      r$n_seeds, r$n_articles_pre, r$n_articles_post,
      r$ai_pct_change, r$ai_p_value, r$ai_p_adjusted,
      r$baseline_pct_change, r$baseline_p_value, r$baseline_p_adjusted
    ))
  }

  all_results
}

plot_split_corpus <- function(split_corpus_df, save = TRUE) {
  # Mirrors Figure 2's visual contrast: AI-associated top-N words vs
  # matched baseline words, pre/post % change, faceted by N, one point per
  # outlet (our analogue of "by language").
  if (is.null(split_corpus_df) || nrow(split_corpus_df) == 0) return(NULL)

  plot_df <- split_corpus_df %>%
    filter(group != "(pooled)") %>%
    dplyr::select(group, n_seeds, ai_pct_change, baseline_pct_change) %>%
    pivot_longer(cols = c(ai_pct_change, baseline_pct_change),
                 names_to = "word_set", values_to = "pct_change") %>%
    mutate(word_set = recode(word_set,
                             ai_pct_change = "AI-associated (top-N)",
                             baseline_pct_change = "Matched baseline (|LPR| ~ 0)")) %>%
    filter(!is.na(pct_change))

  if (nrow(plot_df) == 0) return(NULL)

  p <- ggplot(plot_df, aes(x = group, y = pct_change, colour = word_set, shape = word_set)) +
    geom_point(size = 3, position = position_dodge(width = 0.5)) +
    geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50") +
    facet_wrap(vars(n_seeds), labeller = labeller(n_seeds = function(x) paste0("top-", x))) +
    scale_colour_manual(values = c("AI-associated (top-N)" = unname(CB_PALETTE["vermillion"]),
                                   "Matched baseline (|LPR| ~ 0)" = unname(CB_PALETTE["blue"]))) +
    labs(
      title = "Pre/post prevalence change: AI-associated vs matched baseline words",
      subtitle = paste0("Split at ", format(INTERVENTION_DATE, "%B %Y"),
                        " (Juzek 2026 Sec. 3.5 design); rate per 1000 words, by outlet"),
      x = NULL, y = "% change in prevalence (pre to post)", colour = NULL, shape = NULL
    ) +
    theme_thesis() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))

  if (save) {
    dir.create(PLOT_DIR, recursive = TRUE, showWarnings = FALSE)
    ggsave(file.path(PLOT_DIR, "split_corpus_pre_post.pdf"), p, width = 10, height = 6)
    message("[plot] Saved: ", file.path(PLOT_DIR, "split_corpus_pre_post.pdf"))
  }
  p
}

# =============================================================================
# 9c. SPLIT-CORPUS PRE/POST ANALYSIS — ALL OTHER METRICS
# =============================================================================
# Same clean, deliberately-separate binary before/after design as Sec. 9b —
# split the corpus at INTERVENTION_DATE and compare the two halves directly,
# kept out of the continuous ITS regression machinery — extended to every
# OTHER metric (fingerprint + semantic-category + composite_fingerprint +
# pangram_ai_score), not just the top-N AI-seed/baseline word counts.
#
# These metrics aren't raw token counts, so Sec. 9b's chi-square design
# ("target word count" vs "all other tokens") doesn't apply to them. Instead
# this runs a two-sample Welch's t-test (unequal variances, no assumption
# the pre/post groups have the same spread) directly on article-level
# values, pre vs post, per outlet and pooled — the natural equivalent of
# Sec. 9b's test for a continuous, already-normalised metric. The top-N
# AI-seed/baseline columns are explicitly excluded here since Sec. 9b
# already tests those (on raw counts, the more appropriate design for them).
# =============================================================================

run_split_corpus_ttest <- function(df, metric, group_col = "outlet",
                                   min_n = 10) {
  if (!metric %in% names(df)) return(NULL)

  run_one <- function(d, label) {
    pre  <- d %>% filter(split_period == "pre")  %>% pull(.data[[metric]]) %>% na.omit()
    post <- d %>% filter(split_period == "post") %>% pull(.data[[metric]]) %>% na.omit()
    if (length(pre) < min_n || length(post) < min_n) return(NULL)

    test <- tryCatch(t.test(post, pre), error = function(e) NULL)  # Welch's by default
    if (is.null(test)) return(NULL)

    pre_mean  <- mean(pre)
    post_mean <- mean(post)

    tibble(
      group      = label,
      metric     = metric,
      pre_mean   = pre_mean,
      post_mean  = post_mean,
      pct_change = if (pre_mean != 0) (post_mean - pre_mean) / abs(pre_mean) * 100 else NA_real_,
      t_statistic = unname(test$statistic),
      p_value     = test$p.value,
      n_pre  = length(pre),
      n_post = length(post)
    )
  }

  pieces <- list("(pooled)" = run_one(df, "(pooled)"))
  if (!is.null(group_col) && group_col %in% names(df)) {
    for (g in sort(unique(df[[group_col]]))) {
      pieces[[g]] <- run_one(df %>% filter(.data[[group_col]] == g), g)
    }
  }
  bind_rows(pieces)
}

run_split_corpus_analysis_continuous <- function(df, metrics = NULL,
                                                  intervention_date = INTERVENTION_DATE,
                                                  p_adjust = P_ADJUST_METHOD) {
  message("\n=== SPLIT-CORPUS PRE/POST ANALYSIS — OTHER METRICS (Sec. 9c) ===")
  message("[split-corpus-continuous] Same clean before/after split as Sec. 9b, ",
          "extended to every other metric via Welch's t-test on article-level ",
          "values (chi-square only applies to the count-based AI-seed/",
          "baseline word metrics, which keep their Sec. 9b treatment).")

  if (is.null(metrics)) {
    metrics <- c(ALL_METRICS, "composite_fingerprint")
    if ("pangram_ai_score" %in% names(df)) metrics <- c(metrics, "pangram_ai_score")
  }
  # Exclude the top-N AI-seed/baseline columns -- already tested in Sec. 9b.
  focal_pattern <- paste(.AI_SEED_RATIO_RE, .AI_SEED_WEIGHTED_RE,
                         .BASELINE_RATIO_RE, .BASELINE_WEIGHTED_RE, sep = "|")
  metrics <- metrics[!str_detect(metrics, focal_pattern)]
  metrics <- metrics[metrics %in% names(df)]

  if (length(metrics) == 0) {
    message("[split-corpus-continuous] No applicable metrics found.")
    return(NULL)
  }

  d <- split_corpus_period(df, intervention_date)

  all_results <- map_dfr(metrics, function(m) run_split_corpus_ttest(d, m))
  if (nrow(all_results) == 0) {
    message("[split-corpus-continuous] No results produced (groups too small ",
            "on both sides of the split for every metric).")
    return(NULL)
  }

  directions <- get_directional_hypotheses(metrics)
  all_results <- all_results %>%
    mutate(expected_direction = unname(directions[metric])) %>%
    group_by(metric) %>%
    mutate(p_adjusted = p.adjust(p_value, method = p_adjust)) %>%
    ungroup() %>%
    mutate(
      direction_matches = (sign(post_mean - pre_mean) == expected_direction) |
                          (expected_direction == 0),
      significant = p_adjusted < 0.05,
      significant_correct_direction = significant & direction_matches & expected_direction != 0
    )

  fname <- file.path(PLOT_DIR, "split_corpus_pre_post_continuous.csv")
  write_csv(all_results, fname)
  message("[split-corpus-continuous] Wrote results to: ", fname)

  pooled_hits <- all_results %>%
    filter(group == "(pooled)", significant_correct_direction) %>%
    arrange(p_adjusted)
  if (nrow(pooled_hits) > 0) {
    message("[split-corpus-continuous] Pooled metrics significant in the ",
            "hypothesised direction:")
    for (i in seq_len(nrow(pooled_hits))) {
      r <- pooled_hits[i, ]
      message(sprintf("  %s: %+.1f%% (p.adj=%.4g)", r$metric, r$pct_change, r$p_adjusted))
    }
  }

  all_results
}

plot_split_corpus_continuous_heatmap <- function(split_corpus_continuous_df,
                                                  save = TRUE) {
  # Same visual language as plot_results_heatmap() (Sec. 13.5) — metric x
  # outlet tiles, blue = consistent with the AI-adoption hypothesis,
  # vermillion = significant but in the wrong direction.
  if (is.null(split_corpus_continuous_df) || nrow(split_corpus_continuous_df) == 0)
    return(NULL)

  d <- split_corpus_continuous_df %>%
    filter(group != "(pooled)") %>%
    mutate(
      fill_val = case_when(
        significant_correct_direction ~ "Significant\n(correct direction)",
        significant                   ~ "Significant\n(wrong direction)",
        direction_matches %in% TRUE   ~ "Not significant\n(correct direction)",
        TRUE                          ~ "Not significant\n(wrong direction)"
      ),
      fill_val = factor(fill_val, levels = c(
        "Significant\n(correct direction)",
        "Not significant\n(correct direction)",
        "Significant\n(wrong direction)",
        "Not significant\n(wrong direction)"
      )),
      metric_label = sapply(metric, pretty_metric_label)
    )

  fill_colours <- c(
    "Significant\n(correct direction)"     = unname(CB_PALETTE["blue"]),
    "Not significant\n(correct direction)" = unname(CB_PALETTE["sky_blue"]),
    "Significant\n(wrong direction)"       = unname(CB_PALETTE["vermillion"]),
    "Not significant\n(wrong direction)"   = "#D9D9D9"
  )

  p <- ggplot(d, aes(x = group, y = metric_label, fill = fill_val)) +
    geom_tile(colour = "white", linewidth = 0.5) +
    scale_fill_manual(values = fill_colours, name = NULL) +
    labs(
      title    = "Split-corpus pre/post results: all other metrics",
      subtitle = paste0("Welch's t-test, article-level values, split at ",
                        format(INTERVENTION_DATE, "%B %Y"),
                        ". Blue = consistent with AI adoption hypothesis."),
      x = NULL, y = NULL
    ) +
    theme_thesis() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))

  if (save) {
    fname <- file.path(PLOT_DIR, "split_corpus_pre_post_continuous_heatmap.pdf")
    ggsave(fname, p, width = 10, height = 9)
    message("[plot] Saved: ", fname)
  }
  p
}

# =============================================================================
# 9d. STYLISTIC HOMOGENIZATION  (within-outlet + cross-outlet convergence)
# =============================================================================
# Two distinct homogenization questions:
#
#   A. WITHIN-OUTLET: are articles within a publication becoming MORE similar
#      to each other over time? Operationalised as the monthly within-outlet
#      standard deviation of each metric across articles. A declining SD
#      post-2022 = intra-publication homogenization.
#      Method: ITS regression on the SD series (same Newey-West OLS as
#      Sec. 4, but the outcome is SD not mean). Also computed within each
#      outlet × topic cell.
#
#   B. CROSS-OUTLET: are different publications converging toward a common
#      style? Operationalised as the month-by-month standard deviation of
#      outlet-level means across the six outlets. A declining cross-outlet
#      SD post-2022 = inter-publication convergence.
#      Method: Fligner-Killeen non-parametric variance test comparing the
#      distribution of monthly cross-outlet SDs before and after Nov 2022.
# =============================================================================

compute_within_sd <- function(df, metrics = ALL_METRICS,
                               group_cols = c("outlet", "year_month")) {
  metrics <- metrics[metrics %in% names(df)]
  extra   <- intersect(c("outlet_type", "intervention", "time", "time_after"),
                       names(df))
  df %>%
    group_by(across(all_of(c(group_cols, extra)))) %>%
    summarise(
      n_articles = n(),
      across(all_of(metrics),
             ~ sd(.x, na.rm = TRUE),
             .names = "sd_{.col}"),
      .groups = "drop"
    )
}

run_homogenization_its <- function(sd_df, metric,
                                    group_col = "outlet",
                                    min_obs = 15) {
  # ITS on the within-group SD series for one metric.
  # A negative time_after coefficient = SD declining post-intervention
  # = intra-group stylistic homogenization.
  sd_col <- paste0("sd_", metric)
  if (!sd_col %in% names(sd_df)) return(NULL)

  results <- list()
  for (g in sort(unique(sd_df[[group_col]]))) {
    d <- sd_df %>%
      filter(.data[[group_col]] == g, n_articles >= 5, !is.na(.data[[sd_col]]))
    if (nrow(d) < min_obs) next
    tryCatch({
      fit  <- lm(as.formula(paste(sd_col,
                                  "~ time + intervention + time_after + n_articles")),
                 data = d)
      ct   <- coeftest(fit, vcov = NeweyWest(fit, lag = 3))
      ct_df <- as.data.frame(unclass(ct))
      results[[g]] <- tibble(
        group        = g,
        metric       = metric,
        term         = rownames(ct_df),
        estimate     = ct_df[, 1],
        std.error    = ct_df[, 2],
        statistic    = ct_df[, 3],
        p.value      = ct_df[, 4],
        n_obs        = nrow(d),
        r_squared    = summary(fit)$r.squared
      )
    }, error = function(e) NULL)
  }
  if (length(results) == 0) return(NULL)
  bind_rows(results)
}

run_all_homogenization_its <- function(sd_df, metrics = ALL_METRICS,
                                        group_col = "outlet",
                                        p_adjust = P_ADJUST_METHOD) {
  sd_metrics <- intersect(paste0("sd_", metrics), names(sd_df))
  if (length(sd_metrics) == 0) return(NULL)
  raw_metrics <- str_remove(sd_metrics, "^sd_")

  all_results <- map_dfr(raw_metrics,
    ~ run_homogenization_its(sd_df, .x, group_col = group_col))

  if (is.null(all_results) || nrow(all_results) == 0) return(NULL)

  all_results %>%
    group_by(term) %>%
    mutate(p.adjusted = p.adjust(p.value, method = p_adjust)) %>%
    ungroup() %>%
    filter(term %in% c("intervention", "time_after")) %>%
    mutate(homogenizing = (term == "time_after" & estimate < 0) |
                          (term == "intervention" & estimate < 0),
           significant  = p.adjusted < 0.05,
           sig_homogenizing = significant & homogenizing)
}

test_cross_outlet_homogenization <- function(agg_df, metrics = ALL_METRICS,
                                              intervention_date = INTERVENTION_DATE,
                                              p_adjust = P_ADJUST_METHOD) {
  # For each month, compute the SD of outlet means across all outlets.
  # Test whether this cross-outlet variance decreases post-2022 using
  # Fligner-Killeen (non-parametric, robust to non-normality).
  metrics <- metrics[metrics %in% names(agg_df)]

  results <- map_dfr(metrics, function(m) {
    monthly_sd <- agg_df %>%
      filter(!is.na(.data[[m]])) %>%
      group_by(year_month) %>%
      summarise(
        cross_sd    = sd(.data[[m]], na.rm = TRUE),
        n_outlets   = sum(!is.na(.data[[m]])),
        is_post     = any(intervention == 1),
        .groups = "drop"
      ) %>%
      filter(n_outlets >= 3)

    pre  <- monthly_sd$cross_sd[!monthly_sd$is_post]
    post <- monthly_sd$cross_sd[ monthly_sd$is_post]
    if (length(pre) < 5 || length(post) < 5) return(NULL)

    fk  <- fligner.test(c(pre, post),
                         c(rep("pre", length(pre)), rep("post", length(post))))
    wil <- wilcox.test(pre, post, alternative = "greater")  # pre > post = converging

    tibble(
      metric              = m,
      pre_mean_cross_sd   = mean(pre,  na.rm = TRUE),
      post_mean_cross_sd  = mean(post, na.rm = TRUE),
      pct_change          = (mean(post,na.rm=TRUE) - mean(pre,na.rm=TRUE)) /
                              mean(pre, na.rm=TRUE) * 100,
      fligner_statistic   = unname(fk$statistic),
      fligner_p           = fk$p.value,
      wilcoxon_p          = wil$p.value,  # one-sided: pre > post = converging
      n_pre_months        = length(pre),
      n_post_months       = length(post),
      direction           = if_else(mean(post,na.rm=TRUE) < mean(pre,na.rm=TRUE),
                                    "converging", "diverging")
    )
  })

  if (is.null(results) || nrow(results) == 0) return(NULL)

  results %>%
    mutate(
      fligner_p.adjusted  = p.adjust(fligner_p,   method = p_adjust),
      wilcoxon_p.adjusted = p.adjust(wilcoxon_p,  method = p_adjust),
      sig_converging = wilcoxon_p.adjusted < 0.05 & direction == "converging"
    )
}

run_homogenization_analysis <- function(df, agg_df,
                                         metrics = ALL_METRICS,
                                         p_adjust = P_ADJUST_METHOD) {
  message("\n=== STYLISTIC HOMOGENIZATION ANALYSIS (Sec. 9d) ===")
  metrics <- metrics[metrics %in% names(df)]

  # A. Within-outlet: article-level SD per outlet × month
  message("[homog] Computing within-outlet × month SD...")
  sd_outlet <- compute_within_sd(df, metrics,
                                  group_cols = c("outlet", "year_month"))

  its_outlet <- run_all_homogenization_its(sd_outlet, metrics, group_col = "outlet")
  if (!is.null(its_outlet) && nrow(its_outlet) > 0) {
    fname <- file.path(PLOT_DIR, "homogenization_within_outlet_its.csv")
    write_csv(its_outlet, fname)
    message("[homog] Written: ", fname)
    n_sh <- sum(its_outlet$sig_homogenizing, na.rm = TRUE)
    message("[homog] Within-outlet: ", n_sh, " significant homogenizing slopes ",
            "(metric × outlet × term)")
  }

  # A2. Within-outlet × topic
  sd_topic <- NULL
  its_topic <- NULL
  if ("predicted_topic" %in% names(df)) {
    message("[homog] Computing within-outlet × topic × month SD...")
    sd_topic <- compute_within_sd(df, metrics,
                                   group_cols = c("outlet", "predicted_topic",
                                                  "year_month"))
    # ITS on outlet|topic cell label
    sd_topic <- sd_topic %>%
      mutate(.group_label = paste0(outlet, "|", predicted_topic))
    its_topic <- run_all_homogenization_its(sd_topic, metrics,
                                             group_col = ".group_label")
    if (!is.null(its_topic) && nrow(its_topic) > 0) {
      fname <- file.path(PLOT_DIR, "homogenization_within_topic_its.csv")
      write_csv(its_topic, fname)
      message("[homog] Written: ", fname)
    }
  }

  # B. Cross-outlet convergence
  message("[homog] Testing cross-outlet convergence...")
  cross <- test_cross_outlet_homogenization(agg_df, metrics, p_adjust = p_adjust)
  if (!is.null(cross) && nrow(cross) > 0) {
    fname <- file.path(PLOT_DIR, "homogenization_cross_outlet.csv")
    write_csv(cross, fname)
    message("[homog] Written: ", fname)
    n_conv <- sum(cross$sig_converging, na.rm = TRUE)
    message("[homog] Cross-outlet: ", n_conv, "/", nrow(cross),
            " metrics show significant convergence (Wilcoxon, FDR-corrected)")
    message("[homog] Converging metrics: ",
            paste(cross$metric[cross$sig_converging], collapse = ", "))
  }

  invisible(list(
    sd_outlet  = sd_outlet,
    its_outlet = its_outlet,
    sd_topic   = sd_topic,
    its_topic  = its_topic,
    cross      = cross
  ))
}

plot_homogenization_timeline <- function(sd_df, metric,
                                          group_col = "outlet",
                                          unit = "month",
                                          save_path = NULL) {
  sd_col     <- paste0("sd_", metric)
  if (!sd_col %in% names(sd_df)) return(NULL)

  period_col <- if ("year_month" %in% names(sd_df)) "year_month" else
    paste0("period_", unit)

  p <- ggplot(sd_df %>% filter(!is.na(.data[[sd_col]])),
              aes(x      = .data[[period_col]],
                  y      = .data[[sd_col]],
                  colour = .data[[group_col]],
                  fill   = .data[[group_col]])) +
    geom_point(size = PLOT_CONFIG$point_size, alpha = PLOT_CONFIG$point_alpha) +
    geom_smooth(method = "lm", se = TRUE, linewidth = PLOT_CONFIG$line_width,
                alpha = PLOT_CONFIG$ribbon_alpha) +
    geom_vline(xintercept = as.numeric(INTERVENTION_DATE),
               linetype = "dashed", colour = "grey35", linewidth = 0.6) +
    scale_x_date(date_breaks = "3 months", date_labels = "%Y-%m",
                 minor_breaks = NULL) +
    labs(
      title    = paste0("Within-group SD: ", pretty_metric_label(metric)),
      subtitle = "Declining trend = stylistic homogenization (articles becoming more similar)",
      x = NULL, y = paste0("SD of ", pretty_metric_label(metric)),
      colour = group_col, fill = group_col
    ) +
    theme_thesis()

  vals <- unique(na.omit(sd_df[[group_col]]))
  if (all(vals %in% names(OUTLET_IDENT_COLOURS)))
    p <- p + scale_colour_manual(values = OUTLET_IDENT_COLOURS) +
             scale_fill_manual(values = OUTLET_IDENT_COLOURS)

  if (!is.null(save_path)) {
    dir.create(dirname(save_path), recursive = TRUE, showWarnings = FALSE)
    ggsave(save_path, p, width = 9, height = 5)
    message("[plot] Saved: ", save_path)
  }
  invisible(p)
}

# =============================================================================
# 9e. JOURNALIST CONCENTRATION, FLAGGED VS UNFLAGGED, PER-TOPIC PANGRAM
# =============================================================================

# --- 9e.1 Journalist concentration -----------------------------------------
# Are the stylometric shifts concentrated in a small group of journalists,
# or do most journalists shift uniformly? Uses the random time-slope
# estimates from the journalist-level lmer models.
analyze_journalist_concentration <- function(journalist_its_models,
                                              metrics = c("mean_sent_len",
                                                          "nom_rate",
                                                          "mean_dep_depth")) {
  results <- list()
  for (m in metrics) {
    res <- journalist_its_models[[m]]
    if (is.null(res)) next
    # journalist_its$models stores list(model=fit, tidy=..., random=..., metric=...)
    # — extract the actual lmer object from the wrapper list
    lmer_fit <- if (is.list(res) && "model" %in% names(res)) res$model else res
    re <- tryCatch(ranef(lmer_fit)$journalist_id, error = function(e) NULL)
    if (is.null(re) || !"time" %in% colnames(re)) {
      message("[concentration] ", m, ": no 'time' random slope found in ranef()")
      next
    }

    slopes <- re[["time"]]
    n      <- length(slopes)
    if (n < 5) next

    exp_dir <- DIRECTIONAL_HYPOTHESES[m]
    if (is.na(exp_dir)) exp_dir <- 0

    pct_correct <- if (exp_dir < 0) mean(slopes < 0) * 100
                   else if (exp_dir > 0) mean(slopes > 0) * 100
                   else NA_real_

    top10_n    <- max(1L, round(n * 0.1))
    sorted_abs <- sort(abs(slopes), decreasing = TRUE)
    top10_share <- sum(sorted_abs[seq_len(top10_n)]) /
                   sum(sorted_abs) * 100

    results[[m]] <- tibble(
      metric            = m,
      n_journalists     = n,
      mean_slope        = mean(slopes),
      sd_slope          = sd(slopes),
      pct_correct_dir   = pct_correct,
      top10pct_share    = top10_share,
      interpretation    = case_when(
        top10_share > 50 ~ "concentrated (top 10% drive >50% of effect)",
        top10_share > 33 ~ "moderately concentrated",
        TRUE             ~ "broadly distributed"
      )
    )
  }
  if (length(results) == 0) return(NULL)
  bind_rows(results)
}

plot_journalist_concentration <- function(journalist_its_models,
                                           metrics = c("mean_sent_len",
                                                       "nom_rate",
                                                       "mean_dep_depth"),
                                           save = TRUE) {
  plots <- list()
  for (m in metrics) {
    res <- journalist_its_models[[m]]
    if (is.null(res)) next
    lmer_fit <- if (is.list(res) && "model" %in% names(res)) res$model else res
    re <- tryCatch(ranef(lmer_fit)$journalist_id, error = function(e) NULL)
    if (is.null(re) || !"time" %in% colnames(re)) next

    slopes <- re[["time"]]
    d      <- tibble(slope = slopes)
    exp_dir <- DIRECTIONAL_HYPOTHESES[m]
    xint   <- 0
    fill_colour <- unname(CB_PALETTE["blue"])

    p <- ggplot(d, aes(x = slope)) +
      geom_histogram(aes(fill = slope < xint),
                     bins = 30, colour = "white", linewidth = 0.2) +
      geom_vline(xintercept = xint, linetype = "dashed", colour = "grey30") +
      scale_fill_manual(
        values = c("TRUE"  = if (!is.na(exp_dir) && exp_dir < 0)
                                unname(CB_PALETTE["blue"]) else unname(CB_PALETTE["vermillion"]),
                   "FALSE" = "grey75"),
        guide = "none"
      ) +
      labs(
        title    = paste0("Random slopes: ", pretty_metric_label(m)),
        subtitle = paste0("n = ", length(slopes),
                          " journalists. Coloured = expected direction. ",
                          round(mean(slopes < 0) * 100, 1),
                          "% have negative slope."),
        x = "Individual time slope (per month)", y = "Count"
      ) +
      theme_thesis()

    plots[[m]] <- p
    if (save) {
      dir.create(file.path(PLOT_DIR, "diagnostic"), showWarnings = FALSE)
      fname <- file.path(PLOT_DIR, "diagnostic",
                         paste0("concentration_", m, ".pdf"))
      ggsave(fname, p, width = 7, height = 4.5)
      message("[plot] Saved: ", fname)
    }
  }
  invisible(plots)
}

# --- 9e.2 Flagged vs unflagged article comparison --------------------------
compare_flagged_unflagged <- function(df, flagged_files = FLAGGED_FILES,
                                       metrics = FINGERPRINT_METRICS,
                                       p_adjust = P_ADJUST_METHOD) {
  message("\n=== FLAGGED VS UNFLAGGED ARTICLE COMPARISON ===")
  metrics <- metrics[metrics %in% names(df)]
  results <- list()

  for (outlet_code in names(flagged_files)) {
    path <- flagged_files[[outlet_code]]
    if (!file.exists(path)) next

    flagged_raw <- tryCatch(
      jsonlite::fromJSON(path, simplifyDataFrame = TRUE),
      error = function(e) { message("[flagged] Cannot read ", path); NULL }
    )
    if (is.null(flagged_raw)) next

    flagged_dates  <- dmy(flagged_raw$date)
    flagged_titles <- flagged_raw$title

    outlet_df <- df %>% filter(outlet == !!outlet_code)
    join_keys <- data.frame(title = flagged_titles, date = flagged_dates,
                             stringsAsFactors = FALSE)

    flagged_df   <- outlet_df %>%
      inner_join(join_keys, by = c("title", "date"))
    unflagged_df <- outlet_df %>%
      anti_join(join_keys, by = c("title", "date"))

    n_flag <- nrow(flagged_df)
    if (n_flag < 2) {
      message("[flagged] ", outlet_code, ": only ", n_flag,
              " matched flagged articles — skipping.")
      next
    }
    message("[flagged] ", outlet_code, ": ", n_flag,
            " flagged vs ", nrow(unflagged_df), " unflagged.")

    for (m in metrics) {
      fv <- na.omit(flagged_df[[m]])
      uv <- na.omit(unflagged_df[[m]])
      if (length(fv) < 2 || length(uv) < 5) next
      tt <- tryCatch(t.test(fv, uv), error = function(e) NULL)
      if (is.null(tt)) next
      exp_dir <- DIRECTIONAL_HYPOTHESES[m]
      obs_dir <- sign(mean(fv) - mean(uv))
      results[[paste(outlet_code, m)]] <- tibble(
        outlet          = outlet_code,
        outlet_name     = OUTLET_NAMES[outlet_code],
        metric          = m,
        flagged_mean    = mean(fv),
        unflagged_mean  = mean(uv),
        diff            = mean(fv) - mean(uv),
        pct_diff        = (mean(fv) - mean(uv)) / mean(uv) * 100,
        t_statistic     = unname(tt$statistic),
        p_value         = tt$p.value,
        n_flagged       = length(fv),
        n_unflagged     = length(uv),
        expected_direction = if (!is.na(exp_dir)) exp_dir else 0,
        direction_matches  = (!is.na(exp_dir) && exp_dir != 0 &&
                               obs_dir == sign(exp_dir))
      )
    }
  }

  if (length(results) == 0) {
    message("[flagged] No comparison results produced.")
    return(NULL)
  }

  out <- bind_rows(results) %>%
    group_by(metric) %>%
    mutate(p_adjusted = p.adjust(p_value, method = p_adjust)) %>%
    ungroup() %>%
    mutate(significant = p_adjusted < 0.05,
           sig_correct = significant & direction_matches)

  fname <- file.path(PLOT_DIR, "flagged_vs_unflagged_comparison.csv")
  write_csv(out, fname)
  message("[flagged] Written: ", fname)

  sig <- out %>% filter(sig_correct)
  if (nrow(sig) > 0) {
    message("[flagged] Significant correct-direction differences:")
    print(sig %>% dplyr::select(outlet, metric, flagged_mean, unflagged_mean,
                                 pct_diff, p_adjusted))
  }
  invisible(out)
}

plot_flagged_comparison <- function(flagged_comp_df,
                                     metrics = c("mean_sent_len", "nom_rate",
                                                 "mean_dep_depth", "freq_adv"),
                                     save = TRUE) {
  if (is.null(flagged_comp_df) || nrow(flagged_comp_df) == 0) return(NULL)
  metrics <- metrics[metrics %in% flagged_comp_df$metric]

  plot_df <- flagged_comp_df %>%
    filter(metric %in% metrics) %>%
    mutate(
      metric_label = pretty_metric_label(metric),
      group = factor(
        ifelse(sig_correct, "Sig. (correct dir.)",
               ifelse(significant, "Sig. (wrong dir.)", "Not significant")),
        levels = c("Sig. (correct dir.)", "Sig. (wrong dir.)", "Not significant")
      )
    )

  p <- ggplot(plot_df, aes(x = pct_diff, y = metric_label, colour = group)) +
    geom_vline(xintercept = 0, linetype = "dashed", colour = "grey40") +
    geom_point(size = 3) +
    scale_colour_manual(
      values = c("Sig. (correct dir.)"  = unname(CB_PALETTE["blue"]),
                 "Sig. (wrong dir.)"    = unname(CB_PALETTE["vermillion"]),
                 "Not significant"      = "grey65"),
      name = NULL
    ) +
    labs(
      title    = "Flagged vs unflagged articles: stylometric differences",
      subtitle = paste0("% difference (flagged − unflagged). ",
                        "n flagged = ", unique(plot_df$n_flagged), " (DVHN)."),
      x = "% difference (flagged minus unflagged mean)", y = NULL
    ) +
    theme_thesis()

  if (save) {
    fname <- file.path(PLOT_DIR, "flagged_vs_unflagged_dotplot.pdf")
    ggsave(fname, p, width = 9, height = 5)
    message("[plot] Saved: ", fname)
  }
  invisible(p)
}

# --- 9e.3 Per-topic Pangram score (pooled across outlets) ------------------
plot_pangram_by_topic <- function(df, unit = "quarter", save = TRUE) {
  if (!"pangram_ai_score" %in% names(df)) {
    message("[pangram] pangram_ai_score not found — skipping topic plot.")
    return(NULL)
  }
  if (!"predicted_topic" %in% names(df)) {
    message("[pangram] predicted_topic not found — skipping topic plot.")
    return(NULL)
  }

  tl <- build_timeline(df, group_cols = "predicted_topic", unit = unit,
                        metrics = "pangram_ai_score")

  period_col <- paste0("period_", unit)
  p <- ggplot(tl %>% filter(!is.na(pangram_ai_score)),
              aes(x      = .data[[period_col]],
                  y      = pangram_ai_score,
                  colour = predicted_topic,
                  fill   = predicted_topic)) +
    geom_smooth(method = PLOT_CONFIG$trend_method, se = TRUE,
                linewidth = PLOT_CONFIG$line_width,
                alpha = PLOT_CONFIG$ribbon_alpha) +
    geom_point(size = PLOT_CONFIG$point_size, alpha = PLOT_CONFIG$point_alpha) +
    geom_vline(xintercept = as.numeric(INTERVENTION_DATE),
               linetype = "dashed", colour = "grey35", linewidth = 0.6) +
    scale_x_date(date_breaks = "3 months", date_labels = "%Y-%m",
                 minor_breaks = NULL) +
    labs(
      title    = "Pangram AI-assistance score by topic (all outlets pooled)",
      subtitle = paste0(str_to_title(unit),
                        "ly means. Dashed line = November 2022."),
      x = NULL, y = "Mean Pangram AI-assistance score",
      colour = "Topic", fill = "Topic"
    ) +
    theme_thesis()

  if (save) {
    fname <- file.path(PLOT_DIR,
                        paste0("pangram_by_topic_", unit, "ly.pdf"))
    ggsave(fname, p, width = 10, height = 5)
    message("[plot] Saved: ", fname)
  }
  invisible(p)
}

# =============================================================================
# 10. COMPOSITE FINGERPRINT SCORE
# =============================================================================
# Standardise each metric, apply directional sign so all point in the
# same AI-consistent direction, average into one composite score.
# =============================================================================

compute_composite <- function(df, metrics = FINGERPRINT_METRICS) {
  available <- metrics[metrics %in% names(df)]

  directions <- DIRECTIONAL_HYPOTHESES[available]
  available  <- available[directions != 0]    # exclude metrics with no hypothesis
  directions <- directions[available]
  
  message("\n[composite] Building composite fingerprint from ",
          length(available), " metrics: ",
          paste(available, collapse = ", "))
  
  # Standardise each metric (z-score within outlet to remove baseline differences)
  df_scaled <- df %>%
    group_by(outlet) %>%
    mutate(across(all_of(available),
                  ~ as.numeric(scale(.x)),
                  .names = "z_{.col}")) %>%
    ungroup()
  
  # Flip sign so all metrics point in the AI direction (positive = more AI-like)
  for (m in available) {
    col <- paste0("z_", m)
    df_scaled[[col]] <- df_scaled[[col]] * directions[m]
  }
  
  z_cols <- paste0("z_", available)
  df_scaled$composite_fingerprint <- rowMeans(
    df_scaled[, z_cols], na.rm = TRUE
  )
  
  df_scaled
}

# =============================================================================
# 11. SHARED PLOT THEME & COLOURS
# =============================================================================
# ── FONT & SIZE CONTROL ──────────────────────────────────────────────────────
# Change any value here; it propagates to every plot in the script.
PLOT_CONFIG <- list(
  font_family    = "Palatino",   # e.g. "Palatino", "sans", "serif", "Helvetica"
  base_size      = 15,           # overall base font size (pt)
  title_size     = 12,
  subtitle_size  = 9,
  axis_title_size = 15,
  axis_text_size  = 15,
  strip_text_size = 15,
  legend_text_size = 15,
  caption_size   = 12,
  point_size     = 1.5,
  point_alpha    = 0.45,
  line_width     = 1.1,
  ribbon_alpha   = 0.18,
  trend_method   = "lm"          # "lm" or "loess" for the trend line in timeline plots
)
# ─────────────────────────────────────────────────────────────────────────────

theme_thesis <- function() {
  ff <- PLOT_CONFIG$font_family
  theme_minimal(base_size   = PLOT_CONFIG$base_size,
                base_family = ff) +
    theme(
      panel.grid.minor  = element_blank(),
      strip.text        = element_text(face = "bold", size = PLOT_CONFIG$strip_text_size, family = ff),
      legend.position   = "bottom",
      legend.text       = element_text(size = PLOT_CONFIG$legend_text_size, family = ff),
      plot.title        = element_text(face = "bold", size = PLOT_CONFIG$title_size, family = ff),
      plot.subtitle     = element_text(size = PLOT_CONFIG$subtitle_size, colour = "grey40", family = ff),
      axis.title        = element_text(size = PLOT_CONFIG$axis_title_size, colour = "grey20", family = ff),
      axis.text         = element_text(size = PLOT_CONFIG$axis_text_size, colour = "grey20", family = ff),
      axis.text.x       = element_text(angle = 45, hjust = 1,
                                        size = PLOT_CONFIG$axis_text_size, colour = "grey20", family = ff),
      plot.caption      = element_text(size = PLOT_CONFIG$caption_size, colour = "grey50", family = ff)
    )
}

# Colour-blind-safe palette (Okabe & Ito 2008)
CB_PALETTE <- c(
  black          = "#000000",
  orange         = "#E69F00",
  sky_blue       = "#56B4E9",
  bluish_green   = "#009E73",
  yellow         = "#F0E442",
  blue           = "#0072B2",
  vermillion     = "#D55E00",
  reddish_purple = "#CC79A7",
  grey           = "#999999",
  dark_blue      = "#332288",
  light_green    = "#999933",
  dark_green     = "#117733"
)

INTERVENTION_LABEL <- "ChatGPT release\n(30 Nov 2022)"
PERIOD_COLOURS     <- c("Pre-ChatGPT" = unname(CB_PALETTE["sky_blue"]),
                        "Post-ChatGPT" = unname(CB_PALETTE["vermillion"]))
OUTLET_COLOURS     <- c(
  national = unname(CB_PALETTE["blue"]),
  regional = unname(CB_PALETTE["orange"]),
  local    = unname(CB_PALETTE["bluish_green"])
)
# Individual outlet colours — keyed by full display name so legends show
# the full name automatically when colour_col = "outlet_name".
OUTLET_IDENT_COLOURS <- setNames(
  unname(CB_PALETTE[c("sky_blue", "dark_blue", "yellow", "vermillion", "light_green", "dark_green")]),
  unname(OUTLET_NAMES[c("vk","tgf","dvhn","ed","stc","nof")])
)
# Convenience: also available by short code for internal lookups
OUTLET_IDENT_COLOURS_SHORT <- setNames(
  unname(OUTLET_IDENT_COLOURS),
  c("vk","tgf","dvhn","ed","stc","nof")
)

# =============================================================================
# 11b. METRIC LABELS — clean, human-readable axis/title text per metric
# =============================================================================
METRIC_LABELS <- c(
  mtld                   = "MTLD (lexical diversity)",
  prop_low_freq          = "Low-frequency words (%)",
  prop_high_freq         = "High-frequency words (%)",
  mean_zipf              = "Mean word frequency (Zipf)",
  freq_pronoun           = "Pronouns (per 1k words)",
  freq_aux               = "Auxiliary verbs (per 1k words)",
  freq_det               = "Determiners (per 1k words)",
  freq_cconj              = "Coordinating conjunctions (per 1k words)",
  freq_adv               = "Adverbs (per 1k words)",
  freq_adp               = "Adpositions (per 1k words)",
  mean_sent_len          = "Mean sentence length (words)",
  cv_sent_len            = "Sentence-length variability (CV)",
  mean_dep_depth         = "Mean dependency-tree depth",
  finite_verbs_per_sent  = "Finite verbs per sentence",
  mean_tunit_len         = "Mean T-unit length",
  nom_rate               = "Nominalisation rate",
  care_rigour_freq_per1k = "Care/rigour words (per 1k)",
  emphasize_freq_per1k   = "Emphasize words (per 1k)",
  importance_freq_per1k  = "Importance words (per 1k)",
  composite_fingerprint  = "Composite AI-fingerprint score",
  pangram_ai_score       = "Pangram AI-assistance score"
)

pretty_metric_label <- function(metric) {
  vapply(metric, function(m) {
    if (m %in% names(METRIC_LABELS)) return(unname(METRIC_LABELS[m]))
    n <- str_extract(m, "(?<=top)[0-9]+")
    if (!is.na(n)) {
      if (str_detect(m, "^llm_style_word_ratio_top[0-9]+$"))
        return(paste0("AI-seed words, top-", n, " (per 1k)"))
      if (str_detect(m, "^weighted_llm_ratio_top[0-9]+_per1k$"))
        return(paste0("AI-seed words, top-", n, ", weighted (per 1k)"))
      if (str_detect(m, "^baseline_word_ratio_top[0-9]+$"))
        return(paste0("Baseline words, top-", n, " (per 1k)"))
      if (str_detect(m, "^baseline_weighted_ratio_top[0-9]+_per1k$"))
        return(paste0("Baseline words, top-", n, ", weighted (per 1k)"))
    }
    if (str_detect(m, "_freq_per1k$")) {
      cat_name <- str_remove(m, "_freq_per1k$") %>%
        str_replace_all("_", " ") %>% str_to_title()
      return(paste0(cat_name, " words (per 1k)"))
    }
    str_replace_all(m, "_", " ") %>% str_to_title()
  }, character(1), USE.NAMES = FALSE)
}

# =============================================================================
# 11c. PLOT COMBINING HELPERS
# =============================================================================
# assemble_plots() wraps patchwork to lay out a list of ggplot objects.
#
# Usage examples:
#   # All 6 outlets for one metric on one page (2 rows × 3 cols):
#   p_list <- lapply(tl_outlet_list, function(d)
#     plot_timeline(d, "mean_sent_len", unit = "quarterly"))
#   assemble_plots(p_list, ncol = 3, title = "Mean sentence length by outlet")
#
#   # Dual-metric combined view (one metric per row):
#   assemble_plots(list(p_mean_sent, p_nom_rate), ncol = 1)
#
#   # Six-outlet overview matching the example image style:
#   plot_timeline_combined(tl_outlet, "mean_sent_len", group_col = "outlet",
#                          colour_col = "outlet")
# =============================================================================

assemble_plots <- function(plots, ncol = NULL, nrow = NULL,
                            title = NULL, collect_guides = TRUE,
                            save_path = NULL, width = 12, height = 8) {
  # plots: a named or unnamed list of ggplot objects
  result <- patchwork::wrap_plots(plots, ncol = ncol, nrow = nrow)
  if (collect_guides)
    result <- result + patchwork::plot_layout(guides = "collect")
  if (!is.null(title))
    result <- result + patchwork::plot_annotation(
      title = title,
      theme = theme(plot.title = element_text(
        family = PLOT_CONFIG$font_family, face = "bold",
        size = PLOT_CONFIG$title_size
      ))
    )
  if (!is.null(save_path)) {
    dir.create(dirname(save_path), recursive = TRUE, showWarnings = FALSE)
    ggsave(save_path, result, width = width, height = height)
    message("[plot] Saved assembled plot: ", save_path)
  }
  invisible(result)
}

plot_timeline_combined <- function(timeline_df, metric, unit = "month",
                                    group_col = "outlet",
                                    colour_col = "outlet_type",
                                    y_label = NULL, save_path = NULL,
                                    width = 10, height = 5) {
  # All groups (e.g. all 6 outlets) on a single chart — coloured by group or
  # outlet_type. Useful for a quick cross-outlet overview without facets.
  period_col  <- paste0("period_", unit)
  y_lab       <- y_label %||% pretty_metric_label(metric)
  group_sym   <- if (!is.null(group_col)) group_col else colour_col

  p <- ggplot(timeline_df,
              aes(x      = .data[[period_col]],
                  y      = .data[[metric]],
                  colour = .data[[colour_col]],
                  fill   = .data[[colour_col]],
                  group  = .data[[group_sym]])) +
    geom_point(size = PLOT_CONFIG$point_size, alpha = PLOT_CONFIG$point_alpha) +
    geom_smooth(method = PLOT_CONFIG$trend_method, se = TRUE,
                linewidth = PLOT_CONFIG$line_width,
                alpha = PLOT_CONFIG$ribbon_alpha) +
    geom_vline(xintercept = as.numeric(INTERVENTION_DATE),
               linetype = "dashed", colour = "grey35", linewidth = 0.6) +
    scale_x_date(date_breaks = "3 months", date_labels = "%Y-%m",
                 minor_breaks = NULL) +
    labs(title = y_lab, x = NULL, y = y_lab,
         colour = colour_col, fill = colour_col) +
    theme_thesis()

  if (colour_col == "outlet_type")
    p <- p + scale_colour_manual(values = OUTLET_COLOURS) +
             scale_fill_manual(values = OUTLET_COLOURS)
  else if (colour_col == "outlet")
    p <- p + scale_colour_manual(values = OUTLET_IDENT_COLOURS) +
             scale_fill_manual(values = OUTLET_IDENT_COLOURS)

  if (!is.null(save_path)) {
    dir.create(dirname(save_path), recursive = TRUE, showWarnings = FALSE)
    ggsave(save_path, p, width = width, height = height)
  }
  invisible(p)
}

# =============================================================================
# 12. TIMELINE PLOTS — monthly & quarterly, by outlet type / outlet / topic
# =============================================================================
# Each metric × group combination is saved as an individual PDF in a
# metric-named subdirectory (no more faceted multi-group PDFs). Use
# assemble_plots() above to combine plot objects, or
# plot_timeline_combined() for an all-on-one-chart overview.
#
# Folder structure:
#   plots/{monthly|quarterly}/by_outlet_type/{metric}/{outlet_type}.pdf
#   plots/{monthly|quarterly}/by_outlet/{metric}/{outlet}.pdf
#   plots/{monthly|quarterly}/by_topic/{metric}/{outlet}/{safe_topic}.pdf
# =============================================================================

period_floor <- function(date, unit = c("month", "quarter", "year")) {
  unit <- match.arg(unit)
  floor_date(date, unit)
}

build_timeline <- function(df, group_cols, unit = "month", metrics = ALL_METRICS,
                           range_group_cols = group_cols) {
  # One row per group_cols combo x period, for every period between that
  # group's min and max date (zero-filled n_articles, NA metric where there
  # truly are no articles). range_group_cols controls whose date range the
  # skeleton spans -- see the topic-within-outlet call in
  # generate_timeline_plots() for why this is sometimes narrower than
  # group_cols.
  metrics <- metrics[metrics %in% names(df)]
  df <- df %>%
    filter(!is.na(date)) %>%
    mutate(.period = period_floor(date, unit))

  skeleton <- df %>%
    group_by(across(all_of(range_group_cols))) %>%
    summarise(.min = min(.period), .max = max(.period), .groups = "drop") %>%
    rowwise() %>%
    mutate(.period = list(seq(.min, .max, by = unit))) %>%
    ungroup() %>%
    dplyr::select(all_of(range_group_cols), .period) %>%
    unnest(.period)

  if (!setequal(group_cols, range_group_cols)) {
    extra_cols <- setdiff(group_cols, range_group_cols)
    extra_vals <- df %>%
      distinct(across(all_of(unique(c(range_group_cols, extra_cols)))))
    skeleton <- skeleton %>%
      inner_join(extra_vals, by = range_group_cols, relationship = "many-to-many")
  }

  summarised <- df %>%
    group_by(across(all_of(c(group_cols, ".period")))) %>%
    summarise(
      n_articles = n(),
      across(all_of(metrics), ~ mean(.x, na.rm = TRUE)),
      .groups = "drop"
    )

  skeleton %>%
    left_join(summarised, by = c(group_cols, ".period")) %>%
    mutate(n_articles = replace_na(n_articles, 0)) %>%
    rename(!!paste0("period_", unit) := .period)
}


plot_timeline <- function(timeline_df, metric, unit = "month",
                          colour_col = NULL,
                          y_label = NULL, title = NULL,
                          save_path = NULL,
                          width = 7, height = 4.2) {
  # Plots a single-group timeline (one outlet, one outlet_type, or one topic).
  # No facets — call generate_timeline_plots() for the full saved set, or
  # assemble_plots() to lay out multiple returned objects side by side.
  # Data must NOT be pre-filtered on the metric column before passing in;
  # ggplot handles NA gaps on its own while the full date skeleton from
  # build_timeline() keeps the x-axis range correct.
  period_col <- paste0("period_", unit)
  y_lab      <- y_label %||% pretty_metric_label(metric)
  plt_title  <- title   %||% y_lab

  aes_base <- if (!is.null(colour_col))
    aes(x      = .data[[period_col]],
        y      = .data[[metric]],
        colour = .data[[colour_col]],
        fill   = .data[[colour_col]])
  else
    aes(x = .data[[period_col]], y = .data[[metric]])

  p <- ggplot(timeline_df, aes_base) +
    # Trend line + confidence ribbon first (behind points)
    geom_smooth(
      method    = PLOT_CONFIG$trend_method,
      se        = TRUE,
      linewidth = PLOT_CONFIG$line_width,
      alpha     = PLOT_CONFIG$ribbon_alpha
    ) +
    # Raw period-mean points on top
    geom_point(
      size  = PLOT_CONFIG$point_size,
      alpha = PLOT_CONFIG$point_alpha,
      shape = 16
    ) +
    # Intervention marker
    geom_vline(
      xintercept = as.numeric(INTERVENTION_DATE),
      linetype   = "dashed",
      colour     = "grey35",
      linewidth  = 0.6
    ) +
    # Quarterly tick marks regardless of data resolution
    scale_x_date(
      date_breaks  = "3 months",
      date_labels  = "%Y-%m",
      minor_breaks = NULL
    ) +
    labs(
      title    = plt_title,
      subtitle = paste0(str_to_title(unit), "ly means · ",
                        "quarterly x-axis · dashed line = Nov 2022"),
      x = NULL, y = y_lab,
      colour = colour_col, fill = colour_col
    ) +
    theme_thesis()

  # Apply palette when colour_col is a known grouping variable
  if (!is.null(colour_col)) {
    vals <- unique(na.omit(as.character(timeline_df[[colour_col]])))
    if (all(vals %in% names(OUTLET_COLOURS))) {
      p <- p + scale_colour_manual(values = OUTLET_COLOURS,
                                    breaks = OUTLET_TYPE_LEVELS) +
               scale_fill_manual(values = OUTLET_COLOURS,
                                  breaks = OUTLET_TYPE_LEVELS)
    } else if (all(vals %in% names(OUTLET_IDENT_COLOURS))) {
      p <- p + scale_colour_manual(values = OUTLET_IDENT_COLOURS) +
               scale_fill_manual(values = OUTLET_IDENT_COLOURS)
    } else if (all(vals %in% names(OUTLET_IDENT_COLOURS_SHORT))) {
      p <- p + scale_colour_manual(values = OUTLET_IDENT_COLOURS_SHORT) +
               scale_fill_manual(values = OUTLET_IDENT_COLOURS_SHORT)
    }
  }

  if (!is.null(save_path)) {
    dir.create(dirname(save_path), recursive = TRUE, showWarnings = FALSE)
    ggsave(save_path, p, width = width, height = height)
  }
  invisible(p)
}


# Helper: convert a topic label to a safe filename segment
.safe_fname <- function(x) gsub("[^a-zA-Z0-9]+", "_", trimws(x))


generate_timeline_plots <- function(df, metrics = ALL_METRICS,
                                    intervention_date = INTERVENTION_DATE) {
  message("\n=== GENERATING TIMELINE PLOTS (monthly + quarterly × 3 breakdowns) ===")
  metrics   <- metrics[metrics %in% names(df)]
  has_topic <- "predicted_topic" %in% names(df)

  for (unit in TIME_UNITS) {
    unit_dir <- file.path(PLOT_DIR, paste0(unit, "ly"))

    # ── (1) by outlet_type — one file per metric per outlet_type value ──────
    tl_type <- build_timeline(df, group_cols = "outlet_type", unit = unit,
                              metrics = metrics)
    for (m in metrics[metrics %in% names(tl_type)]) {
      for (ot in sort(unique(tl_type$outlet_type))) {
        d <- tl_type %>% filter(outlet_type == !!ot)
        plot_timeline(d, m, unit = unit,
                      title = paste0(pretty_metric_label(m), " — ", str_to_title(ot)),
                      save_path = file.path(unit_dir, "by_outlet_type", m,
                                            paste0(ot, ".pdf")))
      }
    }
    message("[timeline] ", unit, "ly × by_outlet_type: ",
            length(metrics), " metrics × ",
            n_distinct(tl_type$outlet_type), " types = ",
            length(metrics) * n_distinct(tl_type$outlet_type), " files")

    # ── (2) by outlet — one file per metric per outlet ────────────────────
    tl_outlet <- build_timeline(df, group_cols = c("outlet", "outlet_type"),
                                unit = unit, metrics = metrics)
    for (m in metrics[metrics %in% names(tl_outlet)]) {
      for (o in sort(unique(tl_outlet$outlet))) {
        d <- tl_outlet %>% filter(outlet == !!o)
        plot_timeline(d, m, unit = unit, colour_col = "outlet_type",
                      title = paste0(pretty_metric_label(m), " — ",
                                     OUTLET_NAMES[o] %||% o),
                      save_path = file.path(unit_dir, "by_outlet", m,
                                            paste0(o, ".pdf")))
      }
    }
    message("[timeline] ", unit, "ly × by_outlet: ",
            length(metrics), " metrics × ",
            n_distinct(tl_outlet$outlet), " outlets = ",
            length(metrics) * n_distinct(tl_outlet$outlet), " files")

    # ── (3) by topic within outlet — one file per metric × outlet × topic ──
    if (has_topic) {
      tl_topic <- build_timeline(df, group_cols = c("outlet", "predicted_topic"),
                                 unit = unit, metrics = metrics,
                                 range_group_cols = "outlet")
      outlets <- sort(unique(tl_topic$outlet))
      topics  <- sort(unique(tl_topic$predicted_topic))
      n_files <- 0L
      for (m in metrics[metrics %in% names(tl_topic)]) {
        for (o in outlets) {
          for (topic in topics) {
            d <- tl_topic %>% filter(outlet == !!o, predicted_topic == !!topic)
            if (nrow(d) == 0 || all(is.na(d[[m]]))) next
            plot_timeline(d, m, unit = unit,
                          title = paste0(pretty_metric_label(m), " — ",
                                         OUTLET_NAMES[o] %||% o, " / ", topic),
                          save_path = file.path(unit_dir, "by_topic", m, o,
                                                paste0(.safe_fname(topic), ".pdf")))
            n_files <- n_files + 1L
          }
        }
      }
      message("[timeline] ", unit, "ly × by_topic: ", n_files, " files")
    } else {
      message("[timeline] ", unit, "ly × by_topic: skipped (no predicted_topic column)")
    }
  }
  invisible(NULL)
}


# --- 12b. Concept-specific diachronic plot (Juzek 2026 Fig. 3 style) -------
# The three semantic-category metrics (care_rigour/emphasize/importance)
# already get the full generic monthly+quarterly x breakdown treatment via
# generate_timeline_plots() above, like every other metric ("a diachronic
# analysis as everything else"). This function ADDITIONALLY reproduces
# Juzek (2026) Figure 3's specific framing for those three metrics: yearly
# prevalence expressed as % change from each outlet's PRE-intervention
# baseline mean, with per-outlet lines plus a bold cross-outlet mean line —
# the closest analogue this corpus supports to the paper's cross-LANGUAGE
# mean line (we have outlets, not languages). Built on build_timeline()
# (unit="year") so each outlet's line still spans its own full publication
# range with proper zero/NA handling, not just years where the concept
# happens to be non-zero.
plot_concept_diachronic <- function(df, concept_metrics = NULL,
                                    intervention_date = INTERVENTION_DATE,
                                    save = TRUE) {
  if (is.null(concept_metrics)) concept_metrics <- detect_category_metrics(df)
  concept_metrics <- concept_metrics[concept_metrics %in% names(df)]
  if (length(concept_metrics) == 0) {
    message("[concept-diachronic] No category metrics found, skipping.")
    return(NULL)
  }

  yearly <- build_timeline(df, group_cols = "outlet", unit = "year",
                           metrics = concept_metrics)

  baseline_means <- df %>%
    filter(!is.na(date), date < intervention_date) %>%
    group_by(outlet) %>%
    summarise(across(all_of(concept_metrics),
                     ~ mean(.x, na.rm = TRUE),
                     .names = "base_{.col}"),
              .groups = "drop")

  yearly <- yearly %>% left_join(baseline_means, by = "outlet")

  # Compute % change from each outlet's pre-intervention baseline mean, per
  # metric, with plain column arithmetic (not a dynamic rowwise lookup) so
  # the column names stay static and unambiguous.
  for (m in concept_metrics) {
    base_col <- paste0("base_", m)
    pct_col  <- paste0("pct_", m)
    if (base_col %in% names(yearly)) {
      yearly[[pct_col]] <- ifelse(
        !is.na(yearly[[base_col]]) & yearly[[base_col]] > 0,
        (yearly[[m]] - yearly[[base_col]]) / yearly[[base_col]] * 100,
        NA_real_
      )
    }
  }

  plot_df <- yearly %>%
    dplyr::select(outlet, period_year, starts_with("pct_")) %>%
    pivot_longer(cols = starts_with("pct_"), names_to = "metric",
                names_prefix = "pct_", values_to = "pct_change") %>%
    filter(is.finite(pct_change))

  if (nrow(plot_df) == 0) {
    message("[concept-diachronic] No non-missing pct-change values to plot ",
            "(every outlet's pre-intervention baseline mean was 0 or NA).")
    return(NULL)
  }

  mean_line <- plot_df %>%
    group_by(metric, period_year) %>%
    summarise(mean_pct_change = mean(pct_change, na.rm = TRUE), .groups = "drop")

  p <- ggplot() +
    geom_vline(xintercept = intervention_date, linetype = "dashed", colour = "grey40") +
    geom_line(data = plot_df, aes(x = period_year, y = pct_change, group = outlet),
              colour = "grey70", alpha = 0.6, linewidth = 0.4) +
    geom_point(data = plot_df, aes(x = period_year, y = pct_change),
              colour = "grey70", alpha = 0.6, size = 1) +
    geom_line(data = mean_line, aes(x = period_year, y = mean_pct_change),
              colour = "black", linewidth = 1.1) +
    geom_hline(yintercept = 0, linetype = "dotted", colour = "grey50") +
    facet_wrap(vars(metric), scales = "free_y",
              labeller = labeller(metric = pretty_metric_label)) +
    labs(
      title = "Longitudinal prevalence of AI-associated semantic concepts",
      subtitle = paste0("Yearly, % change from each outlet's pre-",
                        format(intervention_date, "%B %Y"),
                        " baseline mean (Juzek 2026 Sec. 3.6 / Fig. 3 design). ",
                        "Thin grey lines = outlets, black = cross-outlet mean."),
      x = NULL, y = "% change from pre-intervention baseline mean"
    ) +
    theme_thesis()

  if (save) {
    dir.create(PLOT_DIR, recursive = TRUE, showWarnings = FALSE)
    ggsave(file.path(PLOT_DIR, "concept_diachronic_pct_change.pdf"), p,
           width = 12, height = 5)
    message("[plot] Saved: ", file.path(PLOT_DIR, "concept_diachronic_pct_change.pdf"))
  }
  p
}


# =============================================================================
# Plots that address a different dimension than the outlet_type/outlet/topic
# breakdown above — individual journalists, model coefficients — and so
# aren't folded into generate_timeline_plots(). Kept at their original
# (monthly-only) granularity; called individually for a handful of key
# metrics in main(), as before.
# =============================================================================

# --- 13.1 ITS time-series plot for a single metric and outlet ---
plot_its_single <- function(agg_df, metric, outlet_name,
                            y_label = NULL, save = TRUE) {
  d <- agg_df %>%
    filter(outlet == outlet_name, !is.na(.data[[metric]])) %>%
    arrange(year_month)
  
  if (nrow(d) == 0) { message("No data for ", outlet_name); return(invisible(NULL)) }
  
  # Fit pre and post regression lines
  pre  <- d %>% filter(intervention == 0)
  post <- d %>% filter(intervention == 1)
  
  formula_str <- paste(metric, "~ time")
  pred_pre  <- if (nrow(pre)  > 2) predict(lm(as.formula(formula_str), pre))  else NULL
  pred_post <- if (nrow(post) > 2) predict(lm(as.formula(formula_str), post)) else NULL
  
  y_lab <- y_label %||% pretty_metric_label(metric)

  p <- ggplot(d, aes(x = year_month, y = .data[[metric]])) +
    geom_point(aes(colour = period), alpha = 0.5, size = 1.5) +
    geom_vline(xintercept = as.numeric(INTERVENTION_DATE),
               linetype = "dashed", colour = "grey30") +
    annotate("text", x = INTERVENTION_DATE, y = Inf,
             label = INTERVENTION_LABEL, hjust = -0.05, vjust = 1.3,
             size = 3, colour = "grey30") +
    scale_colour_manual(values = PERIOD_COLOURS, name = NULL) +
    labs(title    = paste(y_lab, "—", outlet_name),
         subtitle = "Monthly means with ITS regression lines",
         x = NULL, y = y_lab) +
    theme_thesis()
  
  if (!is.null(pred_pre)) {
    p <- p + geom_line(data = pre %>% mutate(.pred = pred_pre),
                       aes(y = .pred), colour = PERIOD_COLOURS[1], linewidth = 1)
  }
  if (!is.null(pred_post)) {
    p <- p + geom_line(data = post %>% mutate(.pred = pred_post),
                       aes(y = .pred), colour = PERIOD_COLOURS[2], linewidth = 1)
  }
  
  if (save) {
    fname <- file.path(PLOT_DIR, paste0("its_", metric, "_", outlet_name, ".pdf"))
    ggsave(fname, p, width = 8, height = 4)
    message("[plot] Saved: ", fname)
  }
  p
}


# --- 13.2 All outlets for one metric (faceted) ---
plot_its_faceted <- function(agg_df, metric, y_label = NULL, save = TRUE) {
  d <- agg_df %>% filter(!is.na(.data[[metric]]))
  y_lab <- y_label %||% pretty_metric_label(metric)
  
  p <- ggplot(d, aes(x = year_month, y = .data[[metric]],
                     colour = outlet_type)) +
    geom_point(alpha = 0.4, size = 1) +
    geom_smooth(aes(group = intervention),
                method = "lm", se = TRUE, linewidth = 0.8) +
    geom_vline(xintercept = as.numeric(INTERVENTION_DATE),
               linetype = "dashed", colour = "grey30") +
    facet_wrap(~ outlet, scales = "free_y", ncol = 3) +
    scale_colour_manual(values = OUTLET_COLOURS, name = "Outlet type") +
    labs(title    = y_lab,
         subtitle = "Monthly means by outlet. Lines: pre/post ITS regression.",
         x = NULL, y = y_lab) +
    theme_thesis()
  
  if (save) {
    dir.create(file.path(PLOT_DIR, "its_faceted"), showWarnings = FALSE)
    fname <- file.path(PLOT_DIR, "its_faceted", paste0(metric, ".pdf"))
    ggsave(fname, p, width = 12, height = 8)
    message("[plot] Saved: ", fname)
  }
  p
}


# --- 13.3 Spaghetti plot: individual journalist trajectories ---
plot_journalist_trajectories <- function(df_journalists, metric,
                                         outlet_name = NULL,
                                         max_journalists = 50,
                                         save = TRUE) {
  d <- df_journalists %>%
    filter(!is.na(.data[[metric]]), !is.na(journalist_id))
  
  if (!is.null(outlet_name)) d <- d %>% filter(outlet == outlet_name)
  
  # Monthly means per journalist
  d_monthly <- d %>%
    group_by(journalist_id, outlet, outlet_type, year_month,
             intervention) %>%
    summarise(metric_mean = mean(.data[[metric]], na.rm = TRUE),
              .groups = "drop")
  
  # Sample journalists if too many
  jids <- unique(d_monthly$journalist_id)
  if (length(jids) > max_journalists) {
    jids <- sample(jids, max_journalists)
    d_monthly <- d_monthly %>% filter(journalist_id %in% jids)
  }
  
  y_lab <- pretty_metric_label(metric)
  title  <- if (!is.null(outlet_name)) paste(y_lab, "—", outlet_name) else y_lab
  
  p <- ggplot(d_monthly,
              aes(x = year_month, y = metric_mean,
                  group = journalist_id, colour = outlet_type)) +
    geom_line(alpha = 0.3, linewidth = 0.4) +
    geom_smooth(aes(group = outlet_type), method = "loess",
                se = FALSE, linewidth = 1.5) +
    geom_vline(xintercept = as.numeric(INTERVENTION_DATE),
               linetype = "dashed", colour = "grey30") +
    scale_colour_manual(values = OUTLET_COLOURS, name = "Outlet type") +
    labs(title    = title,
         subtitle = paste("Individual journalist trajectories (n =",
                          length(jids), "sampled)"),
         x = NULL, y = y_lab) +
    theme_thesis()
  
  if (save) {
    dir.create(file.path(PLOT_DIR, "diagnostic"), showWarnings = FALSE)
    suffix <- if (!is.null(outlet_name)) paste0("_", outlet_name) else ""
    fname  <- file.path(PLOT_DIR, "diagnostic", paste0("trajectories_", metric, suffix, ".pdf"))
    ggsave(fname, p, width = 10, height = 6)
    message("[plot] Saved: ", fname)
  }
  p
}


# --- 13.4 Caterpillar plot: random slopes from lmer ---
plot_random_slopes <- function(lmer_result, metric, save = TRUE) {
  if (is.null(lmer_result)) return(invisible(NULL))
  
  slopes <- lmer_result$random %>%
    filter(term == "time") %>%
    arrange(estimate)
  
  if (nrow(slopes) == 0) return(invisible(NULL))
  
  slopes$journalist_id <- factor(slopes$level,
                                 levels = slopes$level)
  
  y_lab <- paste(pretty_metric_label(metric), "— time slope")
  
  p <- ggplot(slopes, aes(x = estimate, y = journalist_id)) +
    geom_point(size = 0.8, alpha = 0.6) +
    geom_vline(xintercept = 0, linetype = "dashed", colour = "grey40") +
    labs(title    = paste("Random slopes:", y_lab),
         subtitle = "Distribution of individual journalist time trends",
         x = "Random slope estimate", y = NULL) +
    theme_thesis() +
    theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())
  
  if (save) {
    dir.create(file.path(PLOT_DIR, "diagnostic"), showWarnings = FALSE)
    fname <- file.path(PLOT_DIR, "diagnostic", paste0("random_slopes_", metric, ".pdf"))
    ggsave(fname, p, width = 6, height = 5)
    message("[plot] Saved: ", fname)
  }
  p
}


# --- 13.5 Summary heatmap: direction × significance across metrics × outlets ---
plot_results_heatmap <- function(key_coefs_df, term_name = "intervention",
                                 save = TRUE) {
  d <- key_coefs_df %>%
    filter(term == term_name) %>%
    mutate(
      fill_val = case_when(
        significant_correct_direction ~ "Significant\n(correct direction)",
        significant                   ~ "Significant\n(wrong direction)",
        direction_matches %in% TRUE   ~ "Not significant\n(correct direction)",
        TRUE                          ~ "Not significant\n(wrong direction)"
      ),
      fill_val = factor(fill_val, levels = c(
        "Significant\n(correct direction)",
        "Not significant\n(correct direction)",
        "Significant\n(wrong direction)",
        "Not significant\n(wrong direction)"
      )),
      metric_label = sapply(metric, pretty_metric_label)
    )

  # Colour-blind-safe in place of the classic (and unsafe) red/green
  # significance pairing: blue = consistent with the AI-adoption hypothesis,
  # vermillion = significant but in the wrong direction.
  fill_colours <- c(
    "Significant\n(correct direction)"     = unname(CB_PALETTE["blue"]),
    "Not significant\n(correct direction)" = unname(CB_PALETTE["sky_blue"]),
    "Significant\n(wrong direction)"       = unname(CB_PALETTE["vermillion"]),
    "Not significant\n(wrong direction)"   = "#D9D9D9"
  )

  p <- ggplot(d, aes(x = outlet, y = metric_label, fill = fill_val)) +
    geom_tile(colour = "white", linewidth = 0.5) +
    scale_fill_manual(values = fill_colours, name = NULL) +
    labs(title    = paste("ITS results:", term_name, "coefficient"),
         subtitle = "Blue = consistent with AI adoption hypothesis",
         x = NULL, y = NULL) +
    theme_thesis() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))
  
  if (save) {
    fname <- file.path(PLOT_DIR, paste0("heatmap_", term_name, ".pdf"))
    ggsave(fname, p, width = 10, height = 7)
    message("[plot] Saved: ", fname)
  }
  p
}


# --- 13.6 Composite fingerprint trajectory by outlet type ---
plot_composite <- function(df_composite, save = TRUE) {
  d_monthly <- df_composite %>%
    filter(!is.na(composite_fingerprint)) %>%
    group_by(outlet_type, year_month, intervention) %>%
    summarise(mean_composite = mean(composite_fingerprint, na.rm = TRUE),
              se_composite   = sd(composite_fingerprint, na.rm = TRUE) /
                sqrt(n()),
              .groups = "drop")
  
  p <- ggplot(d_monthly,
              aes(x = year_month, y = mean_composite,
                  colour = outlet_type, fill = outlet_type)) +
    geom_ribbon(aes(ymin = mean_composite - se_composite,
                    ymax = mean_composite + se_composite),
                alpha = 0.15, colour = NA) +
    geom_line(linewidth = 1) +
    geom_vline(xintercept = as.numeric(INTERVENTION_DATE),
               linetype = "dashed", colour = "grey30") +
    annotate("text", x = INTERVENTION_DATE, y = Inf,
             label = INTERVENTION_LABEL, hjust = -0.05, vjust = 1.3,
             size = 3, colour = "grey30") +
    scale_colour_manual(values = OUTLET_COLOURS, name = "Outlet type") +
    scale_fill_manual(values = OUTLET_COLOURS,   name = "Outlet type") +
    labs(title    = "Composite AI Fingerprint Score",
         subtitle = "Mean standardised score across all fingerprint metrics\n(positive = more AI-like)",
         x = NULL, y = "Composite score (standardised)") +
    theme_thesis()
  
  if (save) {
    fname <- file.path(PLOT_DIR, "composite_fingerprint.pdf")
    ggsave(fname, p, width = 10, height = 5)
    message("[plot] Saved: ", fname)
  }
  p
}


# --- 13.7 Outlet-type comparison: interaction coefficients forest plot ---
plot_outlet_type_forest <- function(outlet_type_df, metric, save = TRUE) {
  d <- outlet_type_df %>%
    filter(metric == !!metric,
           str_detect(term, "outlet_type|intervention")) %>%
    mutate(
      term_clean = term %>%
        str_replace("outlet_type", "") %>%
        str_replace_all(":", " × ") %>%
        str_trim()
    )
  
  p <- ggplot(d, aes(x = estimate, y = term_clean,
                     xmin = conf.low, xmax = conf.high)) +
    geom_pointrange() +
    geom_vline(xintercept = 0, linetype = "dashed", colour = "grey40") +
    labs(title    = paste("Outlet-type effects:", pretty_metric_label(metric)),
         subtitle = "Mixed-effects model coefficients with 95% CI",
         x = "Coefficient estimate", y = NULL) +
    theme_thesis()
  
  if (save) {
    dir.create(file.path(PLOT_DIR, "diagnostic"), showWarnings = FALSE)
    fname <- file.path(PLOT_DIR, "diagnostic", paste0("outlet_type_forest_", metric, ".pdf"))
    ggsave(fname, p, width = 7, height = 5)
    message("[plot] Saved: ", fname)
  }
  p
}


# --- 13.8 Pangram triangulation plots ---
# Two versions saved automatically:
#   pangram_trajectories_free_scale.pdf  — faceted per outlet, free y-axis,
#     shows per-outlet variation in scale
#   pangram_trajectories_fixed_scale.pdf — all outlets on ONE chart, shared
#     y-axis, coloured by outlet; makes cross-outlet differences visible
# Both use the same scatter + lm trend + confidence-band style as plot_timeline.
plot_pangram <- function(pangram_agg, fixed_scale = FALSE, save = TRUE) {
  if (!"outlet_name" %in% names(pangram_agg)) {
    pangram_agg <- pangram_agg %>%
      mutate(outlet_name = unname(OUTLET_NAMES[outlet]))
  }
  if (!"outlet_type" %in% names(pangram_agg) ||
      !inherits(pangram_agg$outlet_type, "factor")) {
    pangram_agg <- pangram_agg %>%
      mutate(outlet_type = factor(outlet_type, levels = OUTLET_TYPE_LEVELS))
  }

  x_scale <- scale_x_date(date_breaks = "3 months", date_labels = "%Y-%m",
                           minor_breaks = NULL)
  vline    <- geom_vline(xintercept = as.numeric(INTERVENTION_DATE),
                         linetype = "dashed", colour = "grey35", linewidth = 0.6)

  if (fixed_scale) {
    # All outlets on a single chart — coloured by outlet, shared y-axis.
    p <- ggplot(pangram_agg,
                aes(x      = year_month,
                    y      = mean_pangram,
                    colour = outlet_name,
                    fill   = outlet_name)) +
      geom_smooth(method = "lm", se = TRUE,
                  linewidth = PLOT_CONFIG$line_width,
                  alpha = PLOT_CONFIG$ribbon_alpha) +
      geom_point(size = PLOT_CONFIG$point_size, alpha = PLOT_CONFIG$point_alpha) +
      vline + x_scale +
      scale_colour_manual(values = OUTLET_IDENT_COLOURS, name = NULL) +
      scale_fill_manual(  values = OUTLET_IDENT_COLOURS, name = NULL) +
      labs(
        title    = "Pangram AI-assistance score — all outlets, shared scale",
        subtitle = "Monthly means. Trend lines: OLS with 95% CI. Dashed = November 2022.",
        x = NULL, y = "Mean Pangram AI-assistance score"
      ) +
      theme_thesis()

    fname  <- file.path(PLOT_DIR, "pangram_trajectories_fixed_scale.pdf")
    width  <- 10; height <- 5.5

  } else {
    # Faceted per outlet, free y-axis — shows within-outlet trajectory shape.
    p <- ggplot(pangram_agg,
                aes(x      = year_month,
                    y      = mean_pangram,
                    colour = outlet_type,
                    fill   = outlet_type)) +
      geom_smooth(method = "lm", se = TRUE,
                  linewidth = PLOT_CONFIG$line_width,
                  alpha = PLOT_CONFIG$ribbon_alpha) +
      geom_point(size = PLOT_CONFIG$point_size, alpha = PLOT_CONFIG$point_alpha) +
      vline + x_scale +
      facet_wrap(~ outlet_name, ncol = 3, scales = "free_y") +
      scale_colour_manual(values = OUTLET_COLOURS, name = "Outlet type",
                          breaks = OUTLET_TYPE_LEVELS) +
      scale_fill_manual(  values = OUTLET_COLOURS, name = "Outlet type",
                          breaks = OUTLET_TYPE_LEVELS) +
      labs(
        title    = "Pangram AI-assistance score by outlet",
        subtitle = "Monthly means. Trend lines: OLS with 95% CI. Dashed = November 2022. Free y-axis.",
        x = NULL, y = "Mean Pangram AI-assistance score"
      ) +
      theme_thesis()

    fname  <- file.path(PLOT_DIR, "pangram_trajectories_free_scale.pdf")
    width  <- 12; height <- 7
  }

  if (save) {
    ggsave(fname, p, width = width, height = height)
    message("[plot] Saved: ", fname)
  }
  invisible(p)
}

plot_pangram_fraction_bars <- function(df, unit = "month", save = TRUE) {
  # Stacked bar chart of Pangram's AI / AI-assisted / human classification
  # mix over time — the per-article fraction_ai/fraction_ai_assisted/
  # fraction_human columns (article-level, NOT derived from windows),
  # averaged per period per outlet. Complements pangram_ai_score (the
  # continuous mean window score, plotted via the generic timeline system)
  # by showing the categorical breakdown instead.
  frac_cols <- c("pangram_fraction_ai", "pangram_fraction_ai_assisted",
                 "pangram_fraction_human")
  if (!all(frac_cols %in% names(df))) {
    message("[pangram] Fraction columns not found, skipping bar plot.")
    return(NULL)
  }

  period_col <- paste0("period_", unit)
  tl <- build_timeline(df, group_cols = "outlet", unit = unit, metrics = frac_cols)

  plot_df <- tl %>%
    dplyr::select(outlet, all_of(period_col), all_of(frac_cols)) %>%
    pivot_longer(cols = all_of(frac_cols), names_to = "category", values_to = "fraction") %>%
    mutate(
      category = recode(category,
                        pangram_fraction_human       = "Human",
                        pangram_fraction_ai_assisted = "AI-assisted",
                        pangram_fraction_ai          = "AI"),
      category = factor(category, levels = c("Human", "AI-assisted", "AI"))
    )

  bar_width <- switch(unit, month = 27, quarter = 85, 27)

  p <- ggplot(plot_df, aes(x = .data[[period_col]], y = fraction, fill = category)) +
    geom_col(position = "stack", width = bar_width) +
    facet_wrap(vars(outlet), scales = "free_x") +
    scale_fill_manual(values = c("Human" = unname(CB_PALETTE["blue"]),
                                 "AI-assisted" = unname(CB_PALETTE["orange"]),
                                 "AI-generated" = unname(CB_PALETTE["vermillion"]))) +
    labs(
      title    = "Pangram classification mix over time",
      subtitle = paste0(str_to_title(unit), "ly mean fraction per article ",
                        "(AI / AI-assisted / human). Full date range shown ",
                        "even where a period has no matched articles."),
      x = NULL, y = "Mean fraction per article", fill = NULL
    ) +
    theme_thesis()

  if (save) {
    out_dir <- file.path(PLOT_DIR, paste0(unit, "ly"))
    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
    fname <- file.path(out_dir, "pangram_fraction_mix.pdf")
    ggsave(fname, p, width = 12, height = 8)
    message("[plot] Saved: ", fname)
  }
  p
}

# =============================================================================
# 14. MAIN — RUN FULL ANALYSIS
# =============================================================================

main <- function() {

  message("=== LOADING DATA ===")
  df_raw <- load_data(INPUT_FILES, OUTLET_TYPES)
  message("[load] Total rows loaded: ", nrow(df_raw))

  message("\n=== PREPROCESSING ===")
  df <- preprocess(df_raw)

  # Re-derive FOCAL_METRICS / ALL_METRICS / DIRECTIONAL_HYPOTHESES from the
  # focal-word category columns actually present in this data (see Section 0).
  finalize_metric_config(df)

  # Attach Pangram results (pangram_ai_score + fraction_* columns) by
  # (outlet, title, date). No-ops per outlet where no pangram_results file
  # exists; run_pangram_analysis() below detects whether anything was
  # actually attached.
  df <- attach_pangram_results(df)

  message("\n=== QUALITY CHECKS ===")
  quality_checks(df)

  # Outlet-month aggregation for publication-level analysis
  message("\n=== AGGREGATING TO OUTLET-MONTH ===")
  agg <- aggregate_outlet_month(df)

  # Journalist filtering for individual-level analysis
  message("\n=== FILTERING JOURNALISTS ===")
  df_journalists <- filter_journalists(df)

  # Composite fingerprint score (carries the pangram_* columns through
  # untouched, since compute_composite() only adds columns)
  df_composite   <- compute_composite(df)

  # AI-flagged article audit: our own metrics, side by side with Pangram's
  # verdict, for every article Pangram flagged as AI-written/assisted.
  flagged_report <- report_flagged_article_metrics(
    df_composite,
    metrics = c(ALL_METRICS, "composite_fingerprint")
  )

  # --- Run models ---
  outlet_its     <- run_all_outlet_its(agg)
  journalist_its <- run_all_journalist_its(df_journalists)
  outlet_type_results <- run_all_outlet_type_comparisons(df_journalists)
  pangram_results     <- run_pangram_analysis(df)

  # Split-corpus pre/post analysis (Sec. 9b) — deliberately separate from
  # the continuous ITS modelling above; see that section's header comment.
  split_corpus_results <- run_split_corpus_analysis(df_composite)

  # Split-corpus pre/post for all other metrics (Sec. 9c)
  split_corpus_continuous_results <- run_split_corpus_analysis_continuous(df_composite)

  # Stylistic homogenization (Sec. 9d)
  homog_results <- run_homogenization_analysis(df_composite, agg, metrics = ALL_METRICS)

  # --- Save model results to CSV ---
  message("\n=== SAVING MODEL RESULTS ===")
  write_csv(outlet_its$key,
            file.path(PLOT_DIR, "outlet_its_key_coefficients.csv"))
  write_csv(journalist_its$fixed,
            file.path(PLOT_DIR, "journalist_its_fixed_effects.csv"))
  if (!is.null(outlet_type_results) && nrow(outlet_type_results) > 0)
    write_csv(outlet_type_results,
              file.path(PLOT_DIR, "outlet_type_coefficients.csv"))

  message("\n=== ANALYSIS COMPLETE ===")
  message("[output] CSVs saved to: ", PLOT_DIR)
  message("[plots]  Call plot_* functions interactively from the R console.")
  message("         Example: plot_results_heatmap(results$outlet_its$key, 'time_after')")

  # Return all results invisibly for interactive exploration and plotting
  invisible(list(
    df             = df,
    agg            = agg,
    df_journalists = df_journalists,
    df_composite   = df_composite,
    flagged_report = flagged_report,
    outlet_its     = outlet_its,
    journalist_its = journalist_its,
    outlet_type    = outlet_type_results,
    pangram        = pangram_results,
    split_corpus   = split_corpus_results,
    split_corpus_continuous = split_corpus_continuous_results,
    homogenization = homog_results,
    flagged_comp   = flagged_comp
  ))
}

# Run when sourced or executed as a script
# Comment out if you want to run sections interactively
results <- main()