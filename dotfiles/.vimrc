execute pathogen#infect()

set tabstop=4
set softtabstop=4
set shiftwidth=4
set textwidth=79
set expandtab
set autoindent
set fileformat=unix

set history=1000
set ruler

filetype on
filetype plugin indent on
syntax on

set vb t_vb=

set bg=light
hi clear

au BufRead,BufNewFile *.py,*.pyw,*.c,*.h match BadWhitespace /\s\+$/
set encoding=utf-8
