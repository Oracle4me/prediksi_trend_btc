// Chart Js Global State Default
Chart.defaults.color = '#5a729a';
Chart.defaults.font.family = "'Space Mono', monospace";
Chart.defaults.font.size = 10;
Chart.defaults.plugins.legend.position = 'top';
Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(7,11,20,.92)';
Chart.defaults.plugins.tooltip.borderColor = '#1e2d4a';
Chart.defaults.plugins.tooltip.borderWidth = 1;
Chart.defaults.plugins.tooltip.padding = 10;

const COLORS = {
    lstm: '#f7c948',
    mlp: '#f59e0b',
    rf: '#3ecf8e',
    xgb: '#e05b8a',
    actual: '#4f8cff',
    forecast: '#a855f7'
};

let charts = {};
let currentTab = 'lstm';

const FX_RATE = {
    usdToIdr: null,
    source: null,
    lastUpdate: null
};

const STATE = {
    btcHistory: null,
    metrics: null,
    btcForecast: null,
    lstmHistory: null,
    status: null
};


// Helper
function safeNumber(v, fallback = 0) {
    const n = Number(v);
    return Number.isFinite(n) ? n : fallback;
}

function hasValue(v) {
    return v !== null && v !== undefined && !Number.isNaN(Number(v)) && Number.isFinite(Number(v));
}

function rupiah(v) {
    const n = safeNumber(v, 0);

    if (!FX_RATE.usdToIdr) {
        return '$ ' + n.toLocaleString('en-US', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        });
    }

    const idrValue = n * FX_RATE.usdToIdr;

    return 'Rp ' + Math.round(idrValue).toLocaleString('id-ID');
}

function percent(v) {
    return safeNumber(v, 0).toFixed(2) + '%';
}

function r2Format(v) {
    return safeNumber(v, 0).toFixed(4);
}

function percentOrDash(v) {
    return hasValue(v) ? percent(v) : '—';
}

function rupiahOrDash(v) {
    return hasValue(v) ? rupiah(v) : '—';
}

function r2OrDash(v) {
    return hasValue(v) ? r2Format(v) : '—';
}

function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
}

function setClass(id, className) {
    const el = document.getElementById(id);
    if (el) el.className = className;
}

function destroyChart(id) {
    if (charts[id]) {
        charts[id].destroy();
        delete charts[id];
    }
}

function alignSeriesByDates(sourceDates, sourceValues, targetDates) {
    const map = new Map();

    (sourceDates || []).forEach((date, idx) => {
        map.set(date, sourceValues[idx]);
    });

    return (targetDates || []).map(date => {
        const value = map.get(date);
        return value === undefined ? null : value;
    });
}

function getMetric(models, modelName, key) {
    if (!models || !models[modelName]) return null;

    const value = models[modelName][key];

    return hasValue(value) ? Number(value) : null;
}

function bestModelBy(models, key, lowerIsBetter = false) {
    let bestName = null;
    let bestValue = lowerIsBetter ? Infinity : -Infinity;

    for (const [name, data] of Object.entries(models || {})) {
        if (!data || !hasValue(data[key])) continue;

        const value = Number(data[key]);

        if (
            (lowerIsBetter && value < bestValue) ||
            (!lowerIsBetter && value > bestValue)
        ) {
            bestValue = value;
            bestName = name;
        }
    }

    return bestName;
}

function getBestOverallModel(metrics) {
    const models = normalizeModels(metrics);

    const entries = Object.entries(models)
        .filter(([_, data]) =>
            data &&
            hasValue(data.F1_Score) &&
            hasValue(data.Balanced_Accuracy)
        )
        .sort((a, b) => {
            const f1Difference =
                safeNumber(b[1].F1_Score) -
                safeNumber(a[1].F1_Score);

            if (f1Difference !== 0) {
                return f1Difference;
            }

            return (
                safeNumber(b[1].Balanced_Accuracy) -
                safeNumber(a[1].Balanced_Accuracy)
            );
        });

    return entries.length ? entries[0][0] : null;
}

function normalizeModels(metrics) {
    if (!metrics) return {};

    return {
        LSTM: metrics.LSTM,
        MLP: metrics.MLP,
        'Random Forest': metrics.RandomForest,
        XGBoost: metrics.XGBoost
    };
}

function modelColor(name) {
    if (name === 'LSTM') return COLORS.lstm;
    if (name === 'MLP') return COLORS.mlp;
    if (name === 'Random Forest') return COLORS.rf;
    if (name === 'XGBoost') return COLORS.xgb;
    return '#dbeafe';
}

function modelKeyToTitle(key) {
    const map = {
        lstm: 'LSTM',
        mlp: 'MLP',
        rf: 'Random Forest',
        xgb: 'XGBoost'
    };

    return map[key] || key;
}

function modelKeyToColor(key) {
    const map = {
        lstm: COLORS.lstm,
        mlp: COLORS.mlp,
        rf: COLORS.rf,
        xgb: COLORS.xgb
    };

    return map[key] || '#dbeafe';
}


function showPredPlaceholder(message = '⚠️ Data prediksi belum tersedia. Jalankan training terlebih dahulu.') {
    destroyChart('chartPred');

    const canvas = document.getElementById('chartPred');
    if (!canvas) return;

    const parent = canvas.parentElement;
    let placeholder = document.getElementById('predPlaceholder');

    if (!placeholder) {
        placeholder = document.createElement('div');
        placeholder.id = 'predPlaceholder';
        placeholder.className = 'placeholder';
        parent.insertBefore(placeholder, canvas);
    }

    placeholder.innerHTML = `<span>${message}</span>`;
    placeholder.style.display = 'flex';
    canvas.style.display = 'none';
}

function showPredCanvas() {
    const canvas = document.getElementById('chartPred');
    const placeholder = document.getElementById('predPlaceholder');

    if (placeholder) placeholder.style.display = 'none';
    if (canvas) canvas.style.display = 'block';
}

function showForecastPlaceholder(message) {
    const placeholder = document.getElementById('forecastPlaceholder');
    const chartWrap = document.getElementById('forecastChartWrap');

    if (placeholder) {
        placeholder.innerHTML = `<span>${message}</span>`;
        placeholder.style.display = 'flex';
    }

    if (chartWrap) {
        chartWrap.style.display = 'none';
    }
}

function showForecastLoading() {
    const placeholder =
        document.getElementById(
            'forecastPlaceholder'
        );

    const chartWrap =
        document.getElementById(
            'forecastChartWrap'
        );

    if (placeholder) {
        placeholder.innerHTML = `
            <div class="spinner"></div>
            <span>
                Menghitung prediksi trend
                ${STATE.trendHorizon} hari dan estimasi harga
                berbasis trend…
            </span>
        `;

        placeholder.style.display = 'flex';
    }

    if (chartWrap) {
        chartWrap.style.display = 'none';
    }
}

function showForecastCanvas() {
    const placeholder = document.getElementById('forecastPlaceholder');
    const chartWrap = document.getElementById('forecastChartWrap');

    if (placeholder) placeholder.style.display = 'none';
    if (chartWrap) chartWrap.style.display = 'block';
}


// Chart Builder
function makeLineChart(id, datasets, labels, opts = {}) {
    destroyChart(id);

    const canvas = document.getElementById(id);
    if (!canvas) return;

    const ctx = canvas.getContext('2d');

    charts[id] = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    labels: {
                        boxWidth: 10,
                        font: {
                            size: 10
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function (ctx) {
                            if (ctx.parsed.y == null) return '';
                            return ctx.dataset.label + ': ' + rupiah(ctx.parsed.y);
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(30,45,74,.6)'
                    },
                    ticks: {
                        maxTicksLimit: 8,
                        maxRotation: 0
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(30,45,74,.6)'
                    },
                    ticks: {
                        callback: value => rupiah(value)
                    }
                }
            },
            elements: {
                point: {
                    radius: 0,
                    hitRadius: 6
                },
                line: {
                    tension: .35
                }
            },
            ...opts
        }
    });
}


function makeBarChart(id, labels, datasets, opts = {}) {
    destroyChart(id);

    const canvas = document.getElementById(id);
    if (!canvas) return;

    const ctx = canvas.getContext('2d');

    charts[id] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    labels: {
                        boxWidth: 10,
                        font: {
                            size: 10
                        }
                    }
                },
                tooltip: {
                    callbacks: {
                        label: ctx => {
                            if (ctx.parsed.y === null) return null;
                            return ctx.dataset.label + ': ' + safeNumber(ctx.parsed.y).toFixed(2) + '%';
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(30,45,74,.6)'
                    },
                    ticks: {
                        color: '#5a729a',
                        maxRotation: 0
                    }
                },
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: {
                        color: 'rgba(30,45,74,.6)'
                    },
                    ticks: {
                        color: '#5a729a',
                        callback: value => value + '%'
                    },
                    title: {
                        display: true,
                        text: 'Persentase',
                        color: '#5a729a'
                    }
                }
            },
            ...opts
        }
    });
}


// API Layer
async function apiGetJson(url, options = {}) {
    const res = await fetch(url, options);

    let data = null;

    try {
        data = await res.json();
    } catch (_) {
        data = null;
    }

    if (!res.ok || (data && data.error)) {
        throw new Error(data?.error || `Request gagal: ${url}`);
    }

    return data;
}

async function apiIdrRate() {
    return apiGetJson('/api/idr_rate?ts=' + Date.now(), {
        cache: 'no-store'
    });
}

async function apiBtcHistory() {
    return apiGetJson('/api/btc_history');
}

async function apiBtcForecast() {
    return apiGetJson('/api/btc_forecast');
}

async function apiMetrics() {
    return apiGetJson('/api/metrics');
}

async function apiPrediction(model) {
    return apiGetJson('/api/predictions/' + model);
}

async function apiAllPredictions() {
    return apiGetJson('/api/all_predictions');
}

async function apiLstmHistory() {
    return apiGetJson('/api/lstm_history');
}

async function apiStatus() {
    return apiGetJson('/api/status');
}

async function apiStartTraining() {
    return apiGetJson('/api/train', {
        method: 'POST'
    });
}

function apiTrainingLog(onMessage, onDone, onError) {
    const es = new EventSource('/api/train_log');

    es.onmessage = event => {
        try {
            const data = JSON.parse(event.data);

            if (data.done) {
                es.close();
                if (onDone) onDone();
                return;
            }

            if (onMessage) onMessage(data);
        } catch (e) {
            if (onMessage) onMessage(event.data);
        }
    };

    es.onerror = error => {
        es.close();
        if (onError) onError(error);
    };

    return es;
}


// /api/idr_rate
async function loadRate() {
    try {
        const d = await apiIdrRate();

        if (!d || !d.rate) {
            throw new Error('Gagal mengambil kurs USD/IDR');
        }

        FX_RATE.usdToIdr = Number(d.rate);
        FX_RATE.source = d.source || 'unknown';
        FX_RATE.lastUpdate = d.last_update || null;

        console.log(
            `Kurs USD/IDR aktif: ${FX_RATE.usdToIdr} | source: ${FX_RATE.source}`
        );

    } catch (e) {
        console.warn('rate error', e);

        FX_RATE.usdToIdr = null;
        FX_RATE.source = 'unavailable';
        FX_RATE.lastUpdate = null;
    }
}


// /api/btc_history
async function loadHistory() {
    try {
        const d = await apiBtcHistory();

        if (!d || !d.prices || !d.dates) return;

        STATE.btcHistory = d;

        renderBtcPriceFromHistory(d);
        renderHistoryChart(d);

    } catch (e) {
        console.warn('history error', e);
    }
}

function renderBtcPriceFromHistory(d) {
    const prices = d?.prices || [];

    if (!prices.length) {
        setText('btcPrice', '—');
        return;
    }

    const last = Number(prices[prices.length - 1]);
    const prev = Number(prices[prices.length - 2]);

    setText('btcPrice', rupiah(last));

    const chg = prev ? ((last - prev) / prev) * 100 : 0;
    setClass('btcPrice', 'stat-val ' + (chg >= 0 ? 'up' : 'down'));
}

function renderHistoryChart(d) {
    if (!d?.dates || !d?.prices) return;

    makeLineChart('chartHistory', [
        {
            label: 'Harga BTC',
            data: d.prices,
            borderColor: COLORS.lstm,
            backgroundColor: 'rgba(247,201,72,.07)',
            fill: true,
            borderWidth: 2
        }
    ], d.dates);
}


// /api/btc_forecast
async function loadForecast() {
    showForecastLoading();

    try {
        const fc = await apiBtcForecast();

        if (!fc || !fc.dates || !fc.forecast) {
            showForecastPlaceholder('⚠️ Format data forecast tidak valid');
            return;
        }

        STATE.btcForecast = fc;

        if (!STATE.btcHistory) {
            showForecastPlaceholder('⚠️ Data historis belum tersedia. Muat /api/btc_history terlebih dahulu.');
            return;
        }

        renderForecastTrendCard(fc);
        renderForecastChart(STATE.btcHistory, fc);
        renderForecastStats(fc);

        showForecastCanvas();

    } catch (e) {
        console.warn('forecast error', e);
        showForecastPlaceholder('⚠️ Training diperlukan sebelum prediksi');
    }
}

function normalizeForecastTrend(fc) {
    const rawTrend = String(
        fc.trend_class ||
        fc.trend_label ||
        fc.forecast_trend ||
        fc.trend ||
        ''
    ).toLowerCase();

    const probDown = safeNumber(
        fc.prob_downtrend ??
        fc.prob_down ??
        fc.prob_turun,
        null
    );

    const probUp = safeNumber(
        fc.prob_uptrend ??
        fc.prob_up ??
        fc.prob_naik,
        null
    );

    let trendKey = 'bearish';

    if (
        rawTrend.includes('naik') ||
        rawTrend.includes('bull') ||
        rawTrend.includes('up')
    ) {
        trendKey = 'bullish';

    } else if (
        rawTrend.includes('turun') ||
        rawTrend.includes('bear') ||
        rawTrend.includes('down')
    ) {
        trendKey = 'bearish';

    } else if (
        hasValue(probDown) ||
        hasValue(probUp)
    ) {
        trendKey =
            safeNumber(probUp, 0) >=
            safeNumber(probDown, 0)
                ? 'bullish'
                : 'bearish';
    }

    const trendMap = {
        bullish: {
            title: 'Bullish',
            className: 'up',
            icon: 'bi-graph-up-arrow',
            note:
                `Model LSTM memprediksi trend Bitcoin cenderung naik ` +
                `untuk horizon ${STATE.trendHorizon} hari ke depan.`
        },

        bearish: {
            title: 'Bearish',
            className: 'down',
            icon: 'bi-graph-down-arrow',
            note:
                `Model LSTM memprediksi trend Bitcoin cenderung turun ` +
                `untuk horizon ${STATE.trendHorizon} hari ke depan.`
        },

        sideways: {
            title: 'Sideways',
            className: 'neutral',
            icon: 'bi-arrow-left-right',
            note:
                `Model memprediksi Bitcoin cenderung bergerak sideways ` +
                `untuk horizon ${STATE.trendHorizon} hari ke depan.`
        }
    };

    return {
        key: trendKey,
        ...trendMap[trendKey],
        probDown,
        probUp
    };
}

function renderForecastTrendCard(fc) {
    const trendCard = document.getElementById('trendCard');

    if (!trendCard) return;

    const trend = normalizeForecastTrend(fc);

    trendCard.className = `trend-card ${trend.className}`;

    const trendIcon = document.getElementById('trendIcon');
    const trendTitle = document.getElementById('trendTitle');
    const trendNote = document.getElementById('trendNote');
    const trendProb = document.getElementById('trendProb');

    if (trendIcon) {
        trendIcon.innerHTML = `<i class="bi ${trend.icon}"></i>`;
    }

    if (trendTitle) {
        trendTitle.textContent = trend.title;
    }

    if (trendNote) {
        trendNote.textContent = fc.note || trend.note;
    }

    if (trendProb) {
        if (hasValue(trend.probDown) || hasValue(trend.probUp)) {
            trendProb.innerHTML = `
                <span>
                    Turun: ${percentOrDash(trend.probDown)}
                </span>

                <span>
                    Naik: ${percentOrDash(trend.probUp)}
                </span>
            `;
            trendProb.style.display = 'flex';
        } else {
            trendProb.style.display = 'none';
        }
    }

    trendCard.style.display = 'flex';
}

function renderForecastChart(hist, fc) {
    if (!hist?.dates || !hist?.prices || !fc?.dates || !fc?.forecast) return;

    const nextDate = fc.dates[0];
    const nextForecast = Number(fc.forecast[0]);
    const nextUpper = fc.ci_upper && fc.ci_upper.length ? Number(fc.ci_upper[0]) : null;
    const nextLower = fc.ci_lower && fc.ci_lower.length ? Number(fc.ci_lower[0]) : null;

    const lastHistPrice = Number(hist.prices[hist.prices.length - 1]);
    const allDates = [...hist.dates, nextDate];

    const actualSeries = [
        ...hist.prices,
        null
    ];

    const forecastSeries = [
        ...Array(hist.dates.length - 1).fill(null),
        lastHistPrice,
        nextForecast
    ];

    const datasets = [
        {
            label: 'Harga Aktual',
            data: actualSeries,
            borderColor: COLORS.actual,
            backgroundColor: 'rgba(79,140,255,.05)',
            fill: true,
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.35
        },
        {
            label: 'Estimasi Harga Berbasis Prediksi Trend 14 Hari',
            data: forecastSeries,
            borderColor: COLORS.forecast,
            borderDash: [5, 3],
            borderWidth: 2.5,
            pointRadius: 3,
            pointHoverRadius: 5,
            tension: 0.25,
            fill: false
        }
    ];

    if (nextUpper !== null && nextLower !== null) {
        const ciUpper = [
            ...Array(hist.dates.length - 1).fill(null),
            lastHistPrice,
            nextUpper
        ];

        const ciLower = [
            ...Array(hist.dates.length - 1).fill(null),
            lastHistPrice,
            nextLower
        ];

        datasets.unshift(
            {
                label: 'CI Upper (+4%)',
                data: ciUpper,
                borderColor: 'transparent',
                backgroundColor: 'rgba(168,85,247,.12)',
                fill: '+1',
                pointRadius: 0,
                tension: 0.4
            },
            {
                label: 'CI Lower (-4%)',
                data: ciLower,
                borderColor: 'transparent',
                backgroundColor: 'rgba(168,85,247,.12)',
                fill: false,
                pointRadius: 0,
                tension: 0.4
            }
        );
    }

    makeLineChart('chartForecast', datasets, allDates, {
        plugins: {
            legend: {
                labels: {
                    boxWidth: 10,
                    filter: item => !item.text.includes('CI')
                }
            },
            tooltip: {
                callbacks: {
                    label: ctx => {
                        if (ctx.parsed.y === null) return null;
                        return ctx.dataset.label + ': ' + rupiah(ctx.parsed.y);
                    }
                }
            }
        },
        scales: {
            x: {
                grid: {
                    color: 'rgba(30,45,74,.6)'
                },
                ticks: {
                    maxTicksLimit: 10,
                    maxRotation: 0
                }
            },
            y: {
                grid: {
                    color: 'rgba(30,45,74,.6)'
                },
                ticks: {
                    callback: value => rupiah(value)
                }
            }
        }
    });
}

function renderForecastStats(fc) {
    const forecast = Number(fc?.forecast?.[0]);
    const lastActual = Number(fc?.last_actual_price);

    const change = hasValue(fc?.change_pct)
        ? Number(fc.change_pct).toFixed(2)
        : lastActual
            ? (((forecast - lastActual) / lastActual) * 100).toFixed(2)
            : '0.00';

    const isUp = Number(change) >= 0;

    const fstatEnd = document.getElementById('fstatEnd');
    const fstatChg = document.getElementById('fstatChg');
    const forecastStats = document.getElementById('forecastStats');

    const fstatEndLabel = fstatEnd?.closest('.fstat')?.querySelector('.fstat-label');
    const fstatChgLabel = fstatChg?.closest('.fstat')?.querySelector('.fstat-label');

    if (fstatEndLabel) {
        fstatEndLabel.textContent = 'Estimasi Harga Horizon 14 Hari';
    }

    if (fstatChgLabel) {
        fstatChgLabel.textContent = 'Perubahan dari Harga Terakhir';
    }

    if (fstatEnd) {
        fstatEnd.textContent = rupiah(forecast);
        fstatEnd.className = 'fstat-val ' + (isUp ? 'up' : 'down');
    }

    if (fstatChg) {
        fstatChg.textContent = (isUp ? '+' : '') + change + '%';
        fstatChg.className = 'fstat-val ' + (isUp ? 'up' : 'down');
    }

    if (forecastStats) {
        forecastStats.style.display = 'grid';
    }
}


// /api/metrics
async function loadMetrics() {
    try {
        const d = await apiMetrics();

        if (!d) return;

        STATE.metrics = d;

        renderHeroTrendStats(d);
        renderTrendMetricsTable(d);
        renderModelComparisonChart(d);
        renderLstmTrendDetail(d);
        renderTrendVerdict(d);

    } catch (e) {
        console.warn('metrics error', e);
    }
}

function renderHeroTrendStats(metrics) {
    const models = normalizeModels(metrics);
    const lstm = models.LSTM || {};
    const bestModel = getBestOverallModel(metrics);

    setText(
        'lstmBalancedAcc',
        percentOrDash(lstm.Balanced_Accuracy)
    );
    
    setText(
        'lstmF1Score',
        percentOrDash(lstm.F1_Score)
    );

    setText(
        'lstmRecall',
        percentOrDash(lstm.Recall)
    );

    const bestTrendEl = document.getElementById('bestTrendModel');

    if (bestTrendEl) {
        bestTrendEl.textContent = bestModel || '—';
        bestTrendEl.className =
            'stat-val ' + (bestModel === 'LSTM' ? 'gold' : '');
    }
}

function renderTrendMetricsTable(metrics) {
    const models = normalizeModels(metrics);

    const bestOverall = getBestOverallModel(metrics);
    const bestBalanced = bestModelBy(
        models,
        'Balanced_Accuracy',
        false
    );
    const bestPrecision = bestModelBy(
        models,
        'Precision',
        false
    );
    const bestRecall = bestModelBy(
        models,
        'Recall',
        false
    );
    const bestF1 = bestModelBy(
        models,
        'F1_Score',
        false
    );

    const rows = Object.entries(models)
        .filter(([_, data]) => data)
        .map(([name, data]) => ({
            name,
            data,
            color: modelColor(name)
        }))
        .sort((a, b) => {
            const f1Difference =
                safeNumber(b.data.F1_Score) -
                safeNumber(a.data.F1_Score);

            if (f1Difference !== 0) {
                return f1Difference;
            }

            return (
                safeNumber(b.data.Balanced_Accuracy) -
                safeNumber(a.data.Balanced_Accuracy)
            );
        });

    function cell(value, isBest) {
        return `
            <td${isBest ? " class='lstm-best'" : ""}>
                ${percentOrDash(value)}
            </td>
        `;
    }

    let html = `
        <table class="cmp-table">
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Balanced Accuracy ↑</th>
                    <th>Precision ↑</th>
                    <th>Recall ↑</th>
                    <th>F1 Score ↑</th>
                </tr>
            </thead>
            <tbody>
    `;

    rows.forEach(({ name, data, color }, index) => {
        html += `
            <tr>
                <td>
                    <b
                        class="rank-${index + 1}"
                        style="color:${color}"
                    >
                        ${name}${name === bestOverall ? ' ★' : ''}
                    </b>
                </td>

                ${cell(
                    data.Balanced_Accuracy,
                    name === bestBalanced
                )}

                ${cell(
                    data.Precision,
                    name === bestPrecision
                )}

                ${cell(
                    data.Recall,
                    name === bestRecall
                )}

                ${cell(
                    data.F1_Score,
                    name === bestF1
                )}
            </tr>
        `;
    });

    html += `
            </tbody>
        </table>
    `;

    const metricsWrap =
        document.getElementById('metricsWrap');

    if (metricsWrap) {
        metricsWrap.innerHTML = html;
    }
}

function renderModelComparisonChart(metrics) {
    const models = normalizeModels(metrics);

    const labels = [
        'LSTM',
        'MLP',
        'XGBoost',
        'Random Forest'
    ];

    const availableLabels = labels.filter(
        name => models[name]
    );

    if (!availableLabels.length) return;

    makeBarChart(
        'chartModelCompare',
        availableLabels,
        [
            {
                label: 'Balanced Accuracy',
                data: availableLabels.map(
                    name =>
                        safeNumber(
                            models[name]?.Balanced_Accuracy
                        )
                ),
                backgroundColor:
                    'rgba(245,158,11,.30)',
                borderColor: COLORS.mlp,
                borderWidth: 1
            },
            {
                label: 'Precision',
                data: availableLabels.map(
                    name =>
                        safeNumber(
                            models[name]?.Precision
                        )
                ),
                backgroundColor:
                    'rgba(79,140,255,.32)',
                borderColor: COLORS.actual,
                borderWidth: 1
            },
            {
                label: 'Recall',
                data: availableLabels.map(
                    name =>
                        safeNumber(
                            models[name]?.Recall
                        )
                ),
                backgroundColor:
                    'rgba(62,207,142,.32)',
                borderColor: COLORS.rf,
                borderWidth: 1
            },
            {
                label: 'F1 Score',
                data: availableLabels.map(
                    name =>
                        safeNumber(
                            models[name]?.F1_Score
                        )
                ),
                backgroundColor:
                    'rgba(224,91,138,.32)',
                borderColor: COLORS.xgb,
                borderWidth: 1
            }
        ]
    );
}

function renderLstmTrendDetail(metrics) {
    const models = normalizeModels(metrics);
    const lstm = models.LSTM || {};

    const bestBalanced = bestModelBy(
        models,
        'Balanced_Accuracy',
        false
    );
    const bestPrecision = bestModelBy(
        models,
        'Precision',
        false
    );
    const bestRecall = bestModelBy(
        models,
        'Recall',
        false
    );
    const bestF1 = bestModelBy(
        models,
        'F1_Score',
        false
    );

    const lstmMetrics = [
        {
            label: 'Balanced Accuracy',
            val: percentOrDash(
                lstm.Balanced_Accuracy
            ),
            sub:
                'Rata-rata kemampuan mendeteksi trend naik dan turun',
            best: bestBalanced === 'LSTM'
        },
        {
            label: 'Precision',
            val: percentOrDash(
                lstm.Precision
            ),
            sub:
                'Ketepatan saat model memprediksi trend naik',
            best: bestPrecision === 'LSTM'
        },
        {
            label: 'Recall',
            val: percentOrDash(
                lstm.Recall
            ),
            sub:
                'Kemampuan menangkap kondisi trend naik',
            best: bestRecall === 'LSTM'
        },
        {
            label: 'F1 Score',
            val: percentOrDash(
                lstm.F1_Score
            ),
            sub:
                'Keseimbangan antara Precision dan Recall',
            best: bestF1 === 'LSTM'
        }
    ];

    let detailHtml =
        '<div class="metric-row">';

    lstmMetrics.forEach(
        ({ label, val, sub, best }) => {
            detailHtml += `
                <div class="metric-box">
                    <div class="metric-label">
                        ${label}
                    </div>

                    <div class="metric-value ${
                        best ? 'best' : ''
                    }">
                        ${val}
                    </div>

                    <div class="metric-sub">
                        ${sub}
                    </div>
                </div>
            `;
        }
    );

    detailHtml += '</div>';

    const lstmDetail =
        document.getElementById('lstmDetail');

    if (lstmDetail) {
        lstmDetail.innerHTML = detailHtml;
    }
}

function renderTrendVerdict(metrics) {
    const models = normalizeModels(metrics);

    const bestOverall = getBestOverallModel(metrics);

    const bestBalanced = bestModelBy(
        models,
        'Balanced_Accuracy',
        false
    );
    const bestPrecision = bestModelBy(
        models,
        'Precision',
        false
    );
    const bestRecall = bestModelBy(
        models,
        'Recall',
        false
    );
    const bestF1 = bestModelBy(
        models,
        'F1_Score',
        false
    );

    const bestOverallData =
        models[bestOverall] || {};

    const bestBalancedData =
        models[bestBalanced] || {};

    const bestPrecisionData =
        models[bestPrecision] || {};

    const bestRecallData =
        models[bestRecall] || {};

    const bestF1Data =
        models[bestF1] || {};

    const verdictText = `
        <p>
            Berdasarkan hasil evaluasi pada data pengujian,
            model
            <strong
                style="color:${modelColor(bestOverall)}"
            >
                ${bestOverall || '—'}
            </strong>
            dipilih sebagai model terbaik karena memperoleh
            <strong>
                F1 Score tertinggi sebesar
                ${percentOrDash(
                    bestOverallData.F1_Score
                )}
            </strong>.
            Balanced Accuracy digunakan sebagai pembanding
            apabila terdapat model dengan nilai F1 Score
            yang sama.
        </p>

        <p>
            Model
            <strong
                style="color:${modelColor(bestBalanced)}"
            >
                ${bestBalanced || '—'}
            </strong>
            memperoleh
            <strong>
                Balanced Accuracy tertinggi sebesar
                ${percentOrDash(
                    bestBalancedData.Balanced_Accuracy
                )}
            </strong>.
            Nilai ini menunjukkan kemampuan model dalam
            menjaga performa pada kelas trend naik maupun
            trend turun.
        </p>

        <p>
            Dari sisi ketepatan prediksi trend naik,
            model
            <strong
                style="color:${modelColor(bestPrecision)}"
            >
                ${bestPrecision || '—'}
            </strong>
            memiliki
            <strong>
                Precision tertinggi sebesar
                ${percentOrDash(
                    bestPrecisionData.Precision
                )}
            </strong>.

            Sementara itu, model
            <strong
                style="color:${modelColor(bestRecall)}"
            >
                ${bestRecall || '—'}
            </strong>
            memperoleh
            <strong>
                Recall tertinggi sebesar
                ${percentOrDash(
                    bestRecallData.Recall
                )}
            </strong>,
            yang menunjukkan kemampuan terbaik dalam
            menangkap kondisi trend naik.
        </p>

        <p>
            Model
            <strong
                style="color:${modelColor(bestF1)}"
            >
                ${bestF1 || '—'}
            </strong>
            memperoleh
            <strong>
                F1 Score tertinggi sebesar
                ${percentOrDash(
                    bestF1Data.F1_Score
                )}
            </strong>.
            Oleh karena itu, model tersebut memiliki
            keseimbangan terbaik antara Precision dan Recall
            serta digunakan sebagai model utama.
        </p>
    `;

    const verdictTextEl =
        document.getElementById('verdictText');

    const verdictEl =
        document.getElementById('verdict');

    if (verdictTextEl) {
        verdictTextEl.innerHTML = verdictText;
    }

    if (verdictEl) {
        verdictEl.style.display = 'block';
    }
}

// /api/predictions/<model>
async function switchTab(tab, btn) {
    currentTab = tab;

    document.querySelectorAll('.tab-btn').forEach(button => {
        button.classList.remove('active', 't-lstm', 't-rf', 't-xgb');
    });

    if (btn) {
        btn.classList.add('active');

        if (tab === 'lstm') btn.classList.add('t-lstm');
        if (tab === 'rf') btn.classList.add('t-rf');
        if (tab === 'xgb') btn.classList.add('t-xgb');
    }

    if (tab === 'all') {
        await loadAllPred();
    } else {
        await loadPred(tab);
    }
}

async function loadPred(model) {
    try {
        const d = await apiPrediction(model);

        if (!d || !d.dates || !d.actual || !d.predicted) {
            showPredPlaceholder('⚠️ Format data prediksi tidak valid.');
            return;
        }

        renderPredictionChart(model, d);

    } catch (e) {
        console.warn('loadPred error', e);
        showPredPlaceholder('⚠️ Data prediksi model belum tersedia.');
    }
}

function renderPredictionChart(model, d) {
    showPredCanvas();

    const datasets = [
        {
            label: 'Harga Aktual',
            data: d.actual,
            borderColor: COLORS.actual,
            borderWidth: 2,
            backgroundColor: 'rgba(79,140,255,.05)',
            fill: true
        }
    ];

    if (d.predicted && d.predicted.some(v => v !== null)) {
        datasets.push({
            label: modelKeyToTitle(model) + ' — Estimasi Harga Berbasis Trend',
            data: d.predicted,
            borderColor: modelKeyToColor(model),
            borderDash: [4, 2],
            borderWidth: 1.5,
            fill: false
        });
    }

    makeLineChart('chartPred', datasets, d.dates);
}


// /api/all_predictions
async function loadAllPred() {
    try {
        const d = await apiAllPredictions();

        if (!d || !d.lstm || !d.rf || !d.xgb || !d.mlp) {
            showPredPlaceholder('⚠️ Data prediksi LSTM, MLP, RF, atau XGBoost belum lengkap.');
            return;
        }

        if (!d.lstm.dates || !d.lstm.actual || !d.lstm.predicted) {
            showPredPlaceholder('⚠️ Format data LSTM tidak valid.');
            return;
        }

        renderAllPredictionChart(d);

    } catch (e) {
        console.warn('loadAllPred error', e);
        showPredPlaceholder('⚠️ Gagal memuat overlay prediksi. Cek console browser.');
    }
}

function renderAllPredictionChart(d) {
    showPredCanvas();

    const labels = d.lstm.dates;

    const datasets = [
        {
            label: 'Harga Aktual',
            data: d.lstm.actual,
            borderColor: COLORS.actual,
            borderWidth: 2,
            backgroundColor: 'rgba(79,140,255,.05)',
            fill: true
        }
    ];

    if (d.lstm.predicted && d.lstm.predicted.some(v => v !== null)) {
        datasets.push({
            label: 'LSTM — Estimasi Harga Berbasis Trend',
            data: d.lstm.predicted,
            borderColor: COLORS.lstm,
            borderWidth: 1.8,
            borderDash: [4, 2],
            fill: false
        });
    }


    if (d.mlp && d.mlp.predicted && d.mlp.predicted.some(v => v !== null)) {
        datasets.push({
            label: 'MLP — Estimasi Harga Berbasis Trend',
            data: alignSeriesByDates(d.mlp.dates || [], d.mlp.predicted || [], labels),
            borderColor: COLORS.mlp,
            borderWidth: 1.5,
            borderDash: [5, 2],
            fill: false
        });
    }

    if (d.rf && d.rf.predicted && d.rf.predicted.some(v => v !== null)) {
        datasets.push({
            label: 'Random Forest — Estimasi Harga Berbasis Trend',
            data: alignSeriesByDates(d.rf.dates || [], d.rf.predicted || [], labels),
            borderColor: COLORS.rf,
            borderWidth: 1.5,
            borderDash: [6, 3],
            fill: false
        });
    }

    if (d.xgb && d.xgb.predicted && d.xgb.predicted.some(v => v !== null)) {
        datasets.push({
            label: 'XGBoost — Estimasi Harga Berbasis Trend',
            data: alignSeriesByDates(d.xgb.dates || [], d.xgb.predicted || [], labels),
            borderColor: COLORS.xgb,
            borderWidth: 1.5,
            borderDash: [2, 2],
            fill: false
        });
    }

    makeLineChart('chartPred', datasets, labels);
}


// /api/lstm_history
async function loadLSTMHistory() {
    try {
        const d = await apiLstmHistory();

        if (!d || !d.epochs || !d.loss || !d.val_loss) return;

        STATE.lstmHistory = d;

        renderLstmHistoryChart(d);

    } catch (e) {
        console.warn('lstm history error', e);
    }
}

function renderLstmHistoryChart(d) {
    const canvas = document.getElementById('chartLSTMLoss');

    if (!canvas) return;

    destroyChart('chartLSTMLoss');

    const ctx = canvas.getContext('2d');

    charts['chartLSTMLoss'] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: d.epochs,
            datasets: [
                {
                    label: 'Train Loss',
                    data: d.loss,
                    borderColor: COLORS.lstm,
                    borderWidth: 2,
                    fill: false
                },
                {
                    label: 'Val Loss',
                    data: d.val_loss,
                    borderColor: '#ff6b6b',
                    borderWidth: 1.5,
                    borderDash: [4, 2],
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    labels: {
                        boxWidth: 10
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(30,45,74,.6)'
                    },
                    title: {
                        display: true,
                        text: 'Epoch',
                        color: '#5a729a'
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(30,45,74,.6)'
                    },
                    title: {
                        display: true,
                        text: 'Loss',
                        color: '#5a729a'
                    }
                }
            },
            elements: {
                point: {
                    radius: 0,
                    hitRadius: 6
                },
                line: {
                    tension: .3
                }
            }
        }
    });
}


// /api/status
async function loadStatus() {
    try {
        const d = await apiStatus();

        STATE.status = d;

        if (hasValue(d?.trend_horizon)) {
            STATE.trendHorizon =
                Number(d.trend_horizon);
        }

        return d;

    } catch (e) {
        console.warn('status error', e);
        return null;
    }
}

// /api/train + /api/train_log
async function startTraining() {
    const btn = document.getElementById('btnTrain');
    const logWrap = document.getElementById('train-log-wrap');
    const log = document.getElementById('train-log');

    if (btn) {
        btn.disabled = true;
        btn.textContent = 'Training berjalan…';
    }

    if (logWrap) {
        logWrap.style.display = 'block';
    }

    if (log) {
        log.textContent = '';
    }

    try {
        await apiStartTraining();

        apiTrainingLog(
            data => {
                if (log) {
                    log.textContent += data + '\n';
                }

                if (logWrap) {
                    logWrap.scrollTop = logWrap.scrollHeight;
                }
            },
            async () => {
                if (btn) {
                    btn.textContent = 'Training Selesai';
                    btn.disabled = false;
                }

                await loadAll();
            },
            () => {
                if (btn) {
                    btn.textContent = 'Coba Lagi...';
                    btn.disabled = false;
                }
            }
        );

    } catch (e) {
        console.warn('training error', e);

        if (btn) {
            btn.textContent = 'Coba Lagi...';
            btn.disabled = false;
        }
    }
}


// load data
async function loadTrainedData() {
    await loadMetrics();
    await loadPred(currentTab);
    await loadLSTMHistory();
    await loadForecast();
}

async function loadAll() {
    const status = await loadStatus();
    
    await loadRate();
    await loadHistory();

    if (status?.trained) {
        await loadTrainedData();
    }
}

window.addEventListener('DOMContentLoaded', async () => {
    await loadAll();
});