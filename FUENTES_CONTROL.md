# Fuente de datos y conciliación del condominio

Actualización: 11 de noviembre de 2026, America/Santiago.

La única publicación operativa es:
https://docs.google.com/spreadsheets/d/e/2PACX-1vQrn8w2pA3tkwgrRzyT2DxUHhDMQeECSSFnvjR-ve2x1l-Oiv1iYBPq4azmY-F7iQ/pub?output=csv

| Hoja | GID | Uso |
| --- | --- | --- |
| Resumen | 1279265217 | Contexto y revisión de la publicación |
| Ingresos | 1555162048 | Libro de ingresos, IDs ING, nombres de propietarios |
| Cobros GC | 498984169 | Tramos por año y cargo por parcela |
| Proyectos | 965577812 | Aporte esperado por parcela y conciliación del auxiliar de cámaras |
| Mantención parcelas | 1147757578 | Cargos y ajustes por parcela y trabajo |
| Costos | 1064134694 | Egresos y reintegros con signo de origen |
| Deudas proveedores | 707622201 | Presupuestos, abonos y compromisos pendientes |
| Origen y control | 2089994970 | Referencias y controles internos de conciliación |

## Un mismo cálculo en todas las vistas

`condominio_data.py` consulta las ocho hojas y produce un `Snapshot`. General, Ingresos, Costos, Obligaciones, Control de obligaciones y Fuentes y control consumen ese corte. No usan el archivo anterior. La caché dura cinco minutos; Actualizar data del URL consulta todo nuevamente. El identificador de corte es una huella del contenido, común a las vistas y a los PDF.

- Caja calculada = ingresos con monto + costos firmados. Un costo positivo es un reintegro.
- Caja tras proveedores = caja calculada − compromisos pendientes. Es una estimación, no una cartola bancaria certificada.
- GC por parcela = suma de los tramos publicados. Varios tramos del mismo año son válidos. No se infiere la fecha exacta del cambio de cuota.
- Proyecto cámaras = aporte esperado de cada parcela − abonos Proyecto de Ingresos. El auxiliar se compara, nunca se suma nuevamente a caja.
- Mantención = cargos y ajustes del mismo trabajo y parcela. Un crédito de otro trabajo se conserva por separado. Los ajustes de este auxiliar no se agregan a caja.
- Proveedores se separa de los costos pagados y de la deuda de los propietarios.
- Los guiones que representan ceros en el libro de Ingresos se conservan como cero, diferenciados de una celda sin monto.
- IDs duplicados, tramos duplicados o esquemas inválidos detienen el cálculo. Las conciliaciones con diferencia se muestran como Revisar.
- El universo de cobranza continúa siendo las parcelas 17–36. Los nombres se obtienen de los últimos registros fechados de Ingresos/Proyectos de la nueva publicación; no se importa el directorio antiguo.

## Trazabilidad

Las tablas conservan ID de movimiento, fila publicada y fila histórica. Los cargos del estado de cuenta indican la hoja y sus IDs o filas de respaldo. Los PDF de cobranza usan la cartera filtrada e incluyen un anexo de referencias. El PDF de Obligaciones usa el mismo generador sin filtros. Los exportables incluyen corte de datos y enlaces de la publicación activa.

Las observaciones sobre calidad se refieren al conjunto de las fuentes incluso cuando la cartera está filtrada. Las conciliaciones comparan cálculos con referencias internas del mismo archivo.

## Conciliación del corte inicial (anterior a Estado de Costos)

| Indicador | CLP |
| --- | ---: |
| Ingresos | 18.985.000 |
| Egresos netos | 17.543.204 |
| Caja calculada | 1.441.796 |
| Proveedores pendientes | 840.000 |
| Caja tras proveedores | 601.796 |
| GC por parcela | 890.000 |
| Pendiente GC | 470.000 |
| Pendiente cámaras | 75.000 |
| Pendiente mantención | 126.990 |
| Total por cobrar | 671.990 |
| Créditos a favor | 230.000 |

Los valores cambian cuando cambia la publicación; no están fijados en el código.

## Verificación

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m streamlit run condominio.py --server.address 127.0.0.1 --server.port 8502
```

Se verificaron las seis secciones y sus filtros con Streamlit AppTest, las nueve conciliaciones con las hojas publicadas y la generación/renderizado de PDF completos y filtrados. Las pruebas unitarias usan datos sintéticos y no requieren conexión.

## Control financiero de obligaciones

`control_finance.py` deriva los indicadores analíticos del mismo libro conciliado. No cambia cargos ni abonos:

- Cobertura ponderada: abonos aplicables por obligación / cargos positivos. El excedente pagado en una obligación no cubre otra.
- Cargos por cobrar: pendiente seleccionado / cargos positivos seleccionados.
- Concentración C3/C5: deuda de las tres/cinco mayores parcelas / pendiente total seleccionado.
- Pendiente medio: pendiente / número de parcelas con deuda.
- Saldo neto contable: pendiente menos créditos. Se informa separado del objetivo de cobranza.
- Puente: cargos netos − abonos del libro de caja + créditos separados = pendiente a cobrar.
- Histórico de abonos: suma mensual del libro Ingresos para las claves parcela/concepto seleccionadas, sin duplicar movimientos por trabajos de mantención. Meses sin movimientos se incluyen como cero entre el primer y último mes observado. El promedio móvil usa tres meses; al inicio usa los meses disponibles.

El panorama es una tabla financiera con encabezados agrupados, cuenta por parcela, pendientes por concepto, créditos, cobertura y participación. Su ordenación es real y el CSV conserva el corte consultado. La ventana de 12/24 meses o histórico modifica únicamente el gráfico temporal. Las cifras de cartera continúan correspondiendo al total de los filtros principales.

La ficha individual muestra todos los conceptos de la parcela y separa obligaciones, abonos de caja y ajustes de mantención. No se calcula DSO, mora ni antigüedad porque las fuentes no definen vencimientos o imputación histórica de los abonos.

Verificación adicional: 15 pruebas unitarias, filtros vacíos, selección sin cargo positivo, ordenación del panorama, ventanas temporales y gráfico de conciliación. Los gráficos se renderizaron como imágenes locales para revisar etiquetas y escalas.

## Flujo de caja y PDF completo del dashboard

`cash_flow` usa todos los movimientos válidos de Ingresos y Costos, con fechas reales de pago. No incluye cuotas por cobrar, el auxiliar de cámaras ni compromisos pendientes a proveedores como si fueran cobros/pagos. Separa ingresos, reintegros positivos de Costos y pagos brutos negativos de Costos. Flujo neto = ingresos + reintegros − egresos. Caja al cierre = suma acumulada del flujo neto desde el primer registro con una apertura inicial de referencia cero. La apertura de una ventana de 12/24 meses conserva el acumulado anterior; no se reinicia en cero. Los movimientos con monto sin fecha se excluyen del gráfico y se identifican con su efecto neto.

Este gráfico tiene alcance global del condominio y no usa filtros de parcela, concepto o estado; las fuentes no permiten asignar todos los costos por parcela. La pantalla y el PDF declaran ese alcance. La línea de saldo usa un eje diferente al de los flujos mensuales.

El botón bajo los selectores exporta ahora `control_dashboard_pdf.py`: resumen y ocho KPI, concentración, cobertura con porcentajes naranjas legibles, tabla completa de parcelas, flujo global de caja, puente de conciliación e histórico de abonos. Excluye la sección Detalle técnico y estado de cuenta y las secciones posteriores. Respeta filtros de cartera, orden del panorama y ventanas temporales de ambos gráficos. Los gráficos son vectoriales y no requieren Kaleido para exportar el PDF.

La exportación completa revisada ocupa seis páginas A4 apaisadas. Se verificaron exportaciones completas, filtradas y de todo el histórico, su lectura visual y 21 pruebas unitarias. La referencia de caja calculada sigue siendo interna y no sustituye una cartola bancaria.

## Estado de pago: criterio operativo vigente

La columna Estado de Costos es ahora la fuente operativa para separar los movimientos realizados de los compromisos. Esta regla reemplaza el uso del auxiliar Deudas proveedores como KPI principal de proveedores por pagar cuando la publicación incluye Estado.

- Pagado: el monto firmado afecta caja realizada. Los negativos son egresos pagados; los positivos, reintegros realizados.
- No pagado: los costos negativos son obligaciones abiertas. Se incluyen en proveedores por pagar y en el escenario de caja, pero no se contabilizan como pagos efectuados.
- Estado vacío o no reconocido: se muestra para revisión y no se presume pagado.
- Sin columna Estado en un archivo histórico: se conserva el criterio histórico de movimientos realizados; el auxiliar queda como referencia histórica.
- No se suman el auxiliar Deudas proveedores y los Costos No pagado. Sus diferencias se exponen como conciliación específica.

El gráfico de caja agrega barras de compromisos No pagado y una línea discontinua de caja al atenderlos. Caja realizada = apertura + ingresos + reintegros Pagado − egresos Pagado. Escenario = caja realizada − compromisos No pagado acumulados según la fecha registrada. La fecha de un No pagado no acredita pago efectivo. Sin fecha, el compromiso sigue en el KPI y se informa su aplicación únicamente en el último punto del escenario; no se inventa una fecha real. Las ventanas temporales conservan los compromisos anteriores y su apertura ajustada.

Al cambiar un registro a Pagado, su monto sale del compromiso y entra en el movimiento de caja: no se descuenta dos veces. El mismo criterio se usa en General, Costos, Obligaciones, flujo de caja y PDF.

Corte revisado con Estado:

| Indicador | CLP |
| --- | ---: |
| Ingresos recibidos | 19.005.000 |
| Egresos netos pagados | 17.543.204 |
| Caja pagada | 1.461.796 |
| Costos No pagado (dos registros) | 826.990 |
| Caja al pagar pendientes | 634.806 |
| Auxiliar Deudas proveedores (no sumado) | 840.000 |
| Diferencia entre Costos y auxiliar | -13.010 |

Se verificaron 28 pruebas unitarias, las seis secciones con los datos actualizados, los filtros de Estado, la transición a Pagado y los escenarios de pendientes sin fecha o de meses anteriores. El PDF conserva seis páginas e incorpora ambos tipos de egreso y ambas líneas de caja.

## Cabecera de liquidez y cobranza

La cabecera del dashboard y del PDF presenta cuatro indicadores, de izquierda a derecha: Caja al día de hoy, Parcelas con pendientes, Cuentas por cobrar y Caja tras pagar proveedores. La tarjeta de proveedores por pagar reemplaza a Pendiente medio por deudor.

Caja y proveedores son globales; el número de parcelas y las cuentas por cobrar siguen los filtros principales. Caja al día de hoy incluye los ingresos y costos reconocidos Pagado con fecha hasta el día actual de Santiago. Los movimientos futuros se excluyen del indicador; los reconocidos sin fecha se incluyen y se identifican. Caja tras pagar proveedores = caja de hoy − Costos No pagado, sin sumar cuentas por cobrar futuras ni descontar créditos de otras parcelas. `dashboard_headlines` comparte este cálculo entre pantalla y PDF.

El PDF exporta el mismo bloque de cuatro indicadores, el KPI de proveedores, y todo el contenido hasta antes de Detalle técnico y estado de cuenta. Se comprobó la ausencia de los rótulos antiguos, el orden de los cuatro indicadores, el comportamiento bajo filtros y la presentación del PDF de seis páginas. Verificación: 31 pruebas unitarias y pruebas de las seis secciones.

## Informe a la comunidad: diseño editorial y lectura por parcela

La primera página del PDF usa una portada editorial de Tesorería y cobranza. Mantiene los cuatro indicadores principales y muestra cómo se distribuye la caja entre compromisos de proveedores y saldo después de pagarlos, con proporciones calculadas. La situación de cobranza presenta cobertura y porcentaje pendiente; una franja contable conserva los importes de cargos, abonos, aplicación, créditos, concentración y proveedores. La presentación no atribuye resultados de gestión que no estén demostrados por los datos.

La tabla del panorama ubica Total por pagar como primera cifra junto a parcela/propietario, en rojo, negrita y con fondo suave. Incluye separadores de grupos y filas, alineación numérica a la derecha y ceros representados como —. El desglose de lo pendiente se mantiene separado de la cuenta publicada y de los créditos. En pantalla, la identidad y el total por pagar quedan fijos durante el desplazamiento horizontal.

`control_pdf_design.py` concentra esta portada y la tabla del PDF; no cambia los cálculos. Se verificaron las 31 pruebas existentes, las seis secciones y el renderizado de los PDF completos y filtrados. La exportación completa conserva seis páginas y termina antes del detalle técnico individual.

La sección Exposición y concentración se presenta al final del PDF como cierre analítico con diseño propio: total pendiente, concentración de los tres y cinco mayores saldos con sus montos, distribución por parcela y cobertura por concepto con importes por cobrar. La distribución muestra las ocho primeras parcelas y agrupa explícitamente las restantes, sin omitir su saldo. El panorama pasa a la segunda página y se actualiza la numeración de secciones. Se verificaron el informe completo de seis páginas y el filtrado de cinco páginas.
