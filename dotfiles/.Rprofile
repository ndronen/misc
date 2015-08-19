# http://stackoverflow.com/questions/1189759/expert-r-users-whats-in-your-rprofile
#
repositories <- list(
  CRAN="http://cran.rstudio.com",
  bioc="http://www.bioconductor.org/packages/release/bioc")
  #rforge="http://R-Forge.R-project.org")
options(repos=repositories)

if (interactive()) {
  cosine_similarity <- function(x, y) { x %*% y / sqrt(x%*%x * y%*%y) }
  # Workaround for 
  # http://r.789695.n4.nabble.com/Troubles-with-stemming-tm-Snowball-packages-under-MacOS-td4292605.html
  Sys.setenv(NOAWT=TRUE)

  library(colorout)

  color_off <- noColorOut
  color_on <- function() {
    setOutputColors256(normal=39, number=51, string=79, const=75, verbose=FALSE)
    # setOutputColors256(202, 214, 209, 179)
  }
  color_on()

  library(devtools)
  library(testthat)

  test <- function(...) {
    color_off(); try(devtools::test(...)); color_on()
  }
  test_file <- function(...) {
    color_off(); try(testthat::test_file(...)); color_on()
  }
  test_package <- function(...) {
    color_off(); try(testthat::test_package(...)); color_on()
  }

  # Print startup message
  #options(setwidth.verbose=1) 
  # Print error message when unable to set width
  options(setwidth.verbose=2)   
  # Print width value
  #options(setwidth.verbose=3) 

  library(setwidth)
  cd <- setwd
  pwd <- getwd

  use_comic_neue <- function() {
    suppressPackageStartupMessages(library(ggplot2))
    suppressPackageStartupMessages(library(extrafont))
    theme_set(
      theme_bw() +
      theme(
        text=element_text(size=18, family="Comic Neue", face="bold"),
        plot.background=element_blank(),
        panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.border=element_blank()
      )
    )
  }

  open_as_spreadsheet <- function(df, autofilter=TRUE) {
    # autofilter: whether to apply a filter to make sorting and
    # filtering easier.
    # Origin: http://jeromyanglim.tumblr.com/post/33825729070/function-to-view-r-data-frame-in-spreadsheet
    open_command <- switch(Sys.info()[['sysname']],
      Windows='open',
      Linux='xdg-open',
      Darwin='open')
    require(XLConnect)
    temp_file <- paste0(tempfile(), '.xlsx')
    wb <- loadWorkbook(temp_file, create = TRUE)
    createSheet(wb, name = "temp")
    writeWorksheet(wb, df, sheet = "temp", startRow = 1, startCol = 1)
    if (autofilter) setAutoFilter(wb, 'temp', aref('A1', dim(df)))
    saveWorkbook(wb, )
    system(paste(open_command, temp_file))
  }

  # Assumed significance level alpha is 0.05 (which gives c(alpha)=1.36).
  # http://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test#Two-sample_Kolmogorov.E2.80.93Smirnov_test
  ks_is_significant <- function(D, n1, n2=NULL, verbose=TRUE) {
    n2 <- ifelse(is.null(n2), n1, n2)
    threshold <- 1.36 * sqrt((n1+n2)/(n1*n2))
    is_significant <- D > threshold
    if (verbose) {
      cat(paste0("significance level alpha = 0.05, c(alpha) = 1.36, n1 = ", n1, ", n2 =", n2, "\n"))
      cat("D =", D, "threshold =", threshold, "\n")
      cat("distributions are significantly different if D > threshold\n")
      cat("difference between distributions is significant?", is_significant, "\n")
    }
    is_significant
  }

  distributions_are_different_ks <- function(x, y) {
    ks <- ks.test(x, y)
    is_significant <- ks_is_significant(
        ks$statistic, length(x), length(y))
    if (is_significant) {
      cat("distributions are different\n")
    } else {
      cat("distributions are the same\n")
    }
    ks$are_different <- is_significant
    print(ks)
  }

  t_is_significant <- function(p_value) {
    p_value < 0.05
  }

  distributions_are_different_t <- function(x, y, ...) {
    result <- t.test(x, y, ...)
    result$are_different <- t_is_significant(result$p.value)
    result
  }

  fisher_r2z <- function(r) {
    0.5 * (log(1+r) - log(1-r))
  }

  install_colorout <- function() {
    download.file("http://www.lepem.ufc.br/jaa/colorout_1.0-3.tar.gz", destfile = "/tmp/colorout_1.0-3.tar.gz")
    install.packages("colorout_1.0-3.tar.gz", type = "source", repos = NULL)
  }
}

.Last <- function() {
  if (!any(commandArgs()=='--no-readline') && interactive()) {
    require(utils)
    try(savehistory(Sys.getenv("R_HISTFILE")))
  }
}

options(stringsAsFactors=FALSE)
