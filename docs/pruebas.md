# Pruebas

## Backend

Comando:

```bash
cd FastApi
source .venv/bin/activate
pytest -q
```

Resultado ejecutado:

```text
10 passed, 1 warning in 0.71s
```

Cobertura funcional incluida:

- `/health` responde.
- `/predict` acepta payload valido.
- Rechazo de glucosa fuera de rango.
- Rechazo de edad invalida.
- Probabilidad entre 0 y 1.
- `risk_percentage` coincide con `probability`.
- Prediccion guardada en SQLite temporal.
- `/predictions` respeta `limit`.
- Limite excesivo devuelve `422`.
- Eliminacion de historial.
- Modelo no disponible devuelve `503`.

## Frontend

Comandos:

```bash
cd diabetes-app
npm run lint
npm run typecheck
npm test -- --runInBand
```

Resultados ejecutados:

```text
lint: ok
typecheck: ok
Test Suites: 3 passed, 3 total
Tests: 10 passed, 10 total
```

Cobertura funcional incluida:

- Campo requerido.
- Rango invalido.
- Rechazo de infinito.
- Conversion del formulario al payload.
- Error de configuracion/API.
- Error de red.
- Error 422.
- Estado de carga en boton.
- Estado vacio de resultado.
- Renderizado de resultado con porcentaje del backend.

## Evidencias pendientes

El estudiante debe capturar manualmente:

- Swagger en `/docs`.
- App en Expo Go con formulario.
- Validacion local en un campo.
- Pantalla de resultado.
- Historial con registros.
- Confirmacion de borrado de historial.
- Terminal con pruebas backend.
- Terminal con pruebas frontend.
