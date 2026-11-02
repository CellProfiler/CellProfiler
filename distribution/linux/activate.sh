(
    pixbuf_query="$CONDA_PREFIX/bin/gdk-pixbuf-query-loaders"
    if [ -x "$pixbuf_query" ]; then
        for pixbuf_loaders in "$CONDA_PREFIX"/lib/gdk-pixbuf-2.0/*/loaders; do
            [ -d "$pixbuf_loaders" ] || continue
            pixbuf_cache="${pixbuf_loaders%/loaders}/loaders.cache"
            pixbuf_refresh=false
            if [ ! -s "$pixbuf_cache" ] || [ "$pixbuf_loaders" -nt "$pixbuf_cache" ]; then
                pixbuf_refresh=true
            else
                for pixbuf_loader in "$pixbuf_loaders"/*.so; do
                    if [ "$pixbuf_loader" -nt "$pixbuf_cache" ]; then
                        pixbuf_refresh=true
                        break
                    fi
                done
            fi
            if [ "$pixbuf_refresh" = true ]; then
                if [ -w "${pixbuf_cache%/*}" ]; then
                    GDK_PIXBUF_MODULEDIR="$pixbuf_loaders" GDK_PIXBUF_MODULE_FILE="$pixbuf_cache" "$pixbuf_query" --update-cache
                else
                    printf 'Cannot initialize GdkPixbuf loaders in read-only environment: %s\n' "$CONDA_PREFIX" >&2
                fi
            fi
        done
    fi
)
