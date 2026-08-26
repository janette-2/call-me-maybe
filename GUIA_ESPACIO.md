# Guia de espacio en 42 Madrid

## El problema

En 42, tu home (`/home/janrodri`) es una particion de solo **~5 GB** (`/dev/sdb`),
separada de la raiz que tiene **732 GB** (`/dev/mapper/...`).

El comando `df -h /` **miente**: muestra los 732 GB de la raiz, no los 5 GB de tu home.

## Cuando aplicar estos arreglos

Si ves este error al ejecutar `make run` o cualquier comando con `uv`/`pip`:

```
No space left on device (os error 28)
```

**No es un bug de tu codigo.** Es que tu home esta lleno y las herramientas
no tienen donde guardar archivos grandes (venvs, caches, modelos).

Diagnosticar rapido:

```bash
df -h /home/janrodri    # Este es el que importa, NO df -h /
```

Si ves Use% cercano al 100%, hay que redirigir las caches a la raiz.

## Pasos para arreglar

### Paso 1: Persistir los exports en ~/.zshrc

Los `export` sueltos no sobreviven a cerrar el terminal. Hay que anadirlos
a `~/.zshrc` una sola vez:

```bash
cat >> ~/.zshrc << 'EOF'
export UV_PROJECT_ENVIRONMENT=/tmp/call-me-maybe-venv
export UV_CACHE_DIR=/tmp/uv-cache
export HF_HOME=/tmp/huggingface-cache
EOF
source ~/.zshrc
```

### Paso 2: Limpiar caches antiguas y reintentar

```bash
rm -rf .venv ~/.cache/uv ~/.cache/huggingface
make run
```

### Paso 3 (si el modelo no se descargo antes)

Si la primera ejecucion fallo antes de que `HF_HOME` estuviera redirigido,
puede quedar un modelo parcial en home. Borrarlo:

```bash
rm -rf ~/.cache/huggingface
make run
```

## Que redirige cada variable

| Variable | Que redirige | Donde se guarda |
|---|---|---|
| `UV_PROJECT_ENVIRONMENT` | El venv de Python | `/tmp/call-me-maybe-venv` |
| `UV_CACHE_DIR` | La cache de uv | `/tmp/uv-cache` |
| `HF_HOME` | Los modelos de HuggingFace (~1.5 GB) | `/tmp/huggingface-cache` |

- **UV_PROJECT_ENVIRONMENT**: el venv se crea dentro del proyecto (`./.venv`),
  que esta en la particion de 5 GB. Sin esto, transformers + torch no caben.
- **UV_CACHE_DIR**: uv descarga los wheels a `~/.cache/uv/`, tambien en home.
- **HF_HOME**: HuggingFace descarga los modelos a `~/.cache/huggingface/`.
  Qwen3-0.6B pesa ~1.5 GB y sin esto no cabe en home.

## Que占 mas espacio en home

| Directorio | Tamano tipico | Que es | Seguro borrar |
|---|---|---|---|
| `~/.config/google-chrome/` | ~600 MB | Datos de Chrome | No borrar (rompe Chrome) |
| `~/.config/Code/` | ~600 MB | VS Code | No borrar (rompe VS Code) |
| `~/.config/Slack/` | ~400 MB | Slack | No borrar (rompe Slack) |
| `~/.local/share/` | ~750 MB | Datos de apps | Revisar |
| `~/snap/` | ~450 MB | Snapshots de snap | Si, seguro |
| `~/.cache/` | ~130 MB | Caches varias | Si, seguro |
| `~/Downloads/` | variable | Lo que descargues | Revisar antes |

```bash
rm -rf ~/snap ~/.cache/uv
```

## Al terminar el proyecto: limpiar tmp

`/tmp` se limpia al reiniciar en la mayoria de distros, pero no siempre.
Borrar manualmente para recuperar espacio:

```bash
rm -rf /tmp/call-me-maybe-venv /tmp/uv-cache /tmp/huggingface-cache
```

## Referencia rapida

```bash
# Diagnostico
df -h /home/janrodri
du -sh ~/.* 2>/dev/null | sort -rh | head -10

# Persistir exports (una sola vez)
cat >> ~/.zshrc << 'EOF'
export UV_PROJECT_ENVIRONMENT=/tmp/call-me-maybe-venv
export UV_CACHE_DIR=/tmp/uv-cache
export HF_HOME=/tmp/huggingface-cache
EOF
source ~/.zshrc

# Fix rapido (despues de persistir)
rm -rf .venv ~/.cache/uv ~/.cache/huggingface
make run

# Limpiar home
rm -rf ~/snap ~/.cache/uv

# Limpiar tmp al terminar el proyecto
rm -rf /tmp/call-me-maybe-venv /tmp/uv-cache /tmp/huggingface-cache
```
