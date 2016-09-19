# http://stackoverflow.com/questions/1189759/expert-r-users-whats-in-your-rprofile

if (interactive()) {
  install_colorout <- function() {
    library(devtools)
    install_github('jalvesaq/colorout')
  }
  cosine_similarity <- function(x, y) { x %*% y / sqrt(x%*%x * y%*%y) }
  # Workaround for 
  # http://r.789695.n4.nabble.com/Troubles-with-stemming-tm-Snowball-packages-under-MacOS-td4292605.html
  Sys.setenv(NOAWT=TRUE)
  library(devtools)

  library(stringr)
  library(colorout)
  color_off <- noColorOut
  color_on <- function() {
    setOutputColors256(normal=39, number=51, string=79, const=75, verbose=FALSE)
    # setOutputColors256(202, 214, 209, 179)
  }
  color_on()

  test <- function(...) {
    color_off(); try(devtools::test(...)); color_on()
  }
  test_file <- function(...) {
    color_off(); try(testthat::test_file(...)); color_on()
  }
  test_package <- function(...) {
    color_off(); try(testthat::test_package(...)); color_on()
  }
  library(setwidth)
  cd <- setwd
  pwd <- getwd
}

repositories <- list(
  CRAN="http://cran.rstudio.com",
  bioc="http://www.bioconductor.org/packages/release/bioc")
  #rforge="http://R-Forge.R-project.org")

options(repos=repositories)
options(stringsAsFactors=FALSE)

.First <- function() {
  # Print startup message
  #options(setwidth.verbose=1) 

  # Print error message when unable to set width
  options(setwidth.verbose=2)   

  # Print width value
  #options(setwidth.verbose=3) 
}

.Last <- function() {
  if (!any(commandArgs()=='--no-readline') && interactive()) {
    require(utils)
    try(savehistory(Sys.getenv("R_HISTFILE")))
  }
}

tryCatch(
  {
    options(width = as.integer(Sys.getenv("COLUMNS")))
  },
  error=function(err) {
    write("Can't get your terminal width. Put ``export COLUMNS'' in your \
        .bashrc. Or something. Setting width to 120 chars", stderr())
  }
)
