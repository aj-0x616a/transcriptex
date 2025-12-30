-- ~/.config/nvim/lua/plugins/vimtex.lua
return {
	"lervag/vimtex",
	lazy = false,
	init = function()
		-- Viewer (macOS Skim)
		vim.g.vimtex_view_method = "skim"
		vim.g.vimtex_view_skim_sync = 1
		vim.g.vimtex_view_skim_activate = 1

		-- Compiler (latexmk) – continuous mode
		vim.g.vimtex_compiler_method = "latexmk"
		vim.g.vimtex_compiler_continuous = 1

		-- Put *all* build artifacts (aux/log/synctex + initial PDF) into ./build
		vim.g.vimtex_compiler_latexmk = {
			options = {
				"-pdf",
				"-interaction=nonstopmode",
				"-synctex=1",
				"-file-line-error",
				-- IMPORTANT: We explicitly set -outdir=build, but we will create it ourselves.
				"-outdir=build",
			},
		}

		-- Vimscript helpers for reliability (dir creation + moving PDF)
		vim.cmd([[
      function! s:ProjectRoot() abort
        " Prefer VimTeX's notion of project root, else use current file's dir
        if exists('b:vimtex') && has_key(b:vimtex, 'root') && !empty(b:vimtex.root)
          return b:vimtex.root
        endif
        return expand('%:p:h')
      endfunction

      function! s:EnsureDirs() abort
        let l:root      = s:ProjectRoot()
        let l:build_dir = l:root . '/build'
        let l:pdf_dir   = l:root . '/pdf'

        " Create directories if missing
        if !isdirectory(l:build_dir)
          call mkdir(l:build_dir, 'p')
        endif
        if !isdirectory(l:pdf_dir)
          call mkdir(l:pdf_dir, 'p')
        endif
      endfunction

      function! s:MovePdf() abort
        let l:root      = s:ProjectRoot()
        let l:tex_base  = expand('%:t:r')
        let l:build_dir = l:root . '/build'
        let l:pdf_dir   = l:root . '/pdf'
        let l:src_pdf   = l:build_dir . '/' . l:tex_base . '.pdf'
        let l:dst_pdf   = l:pdf_dir   . '/' . l:tex_base . '.pdf'

        " Make sure pdf dir exists (in case compile was triggered before EnsureDirs)
        if !isdirectory(l:pdf_dir)
          call mkdir(l:pdf_dir, 'p')
        endif

        " Move (overwrite if exists to avoid errors)
        if filereadable(l:src_pdf)
          if filereadable(l:dst_pdf)
            call delete(l:dst_pdf)
          endif
          call rename(l:src_pdf, l:dst_pdf)

          " Update VimTeX's out_dir to the final PDF location so views keep working
          if exists('b:vimtex')
            let b:vimtex.out_dir = l:pdf_dir
          endif

          echo 'PDF moved → ' . l:dst_pdf
        endif
      endfunction

      augroup VimtexBuildAndMove
        autocmd!
        " Ensure directories exist whenever we open TeX or start a compile
        autocmd FileType tex call s:EnsureDirs()
        autocmd User VimtexEventCompileStarted call s:EnsureDirs()

        " Move PDF only after a successful compile
        autocmd User VimtexEventCompileSuccess call s:MovePdf()
      augroup END
    ]])
	end,
}
