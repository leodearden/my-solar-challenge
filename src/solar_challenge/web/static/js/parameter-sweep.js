document.addEventListener('alpine:init', () => {
    Alpine.data('parameterSweep', () => ({
        // Sweep configuration
        parameter: 'pv_capacity_kw',
        minVal: 2.0,
        maxVal: 8.0,
        steps: 5,
        mode: 'linear',

        // Base config
        configMode: 'preset',
        basePreset: '',
        baseConfig: {
            battery_kwh: 5.0,
            location: 'bristol',
            days: 7
        },

        // Results
        sweepId: null,
        sweepValues: [],
        sweepResults: [],
        submitting: false,
        errorMsg: '',

        // Sweep point preview calculations
        get previewValues() {
            const min = parseFloat(this.minVal);
            const max = parseFloat(this.maxVal);
            const n = parseInt(this.steps);
            if (isNaN(min) || isNaN(max) || isNaN(n) || n < 2 || min >= max) return [];

            let values = [];
            if (this.mode === 'geometric') {
                const logMin = Math.log(min);
                const logMax = Math.log(max);
                for (let i = 0; i < n; i++) {
                    values.push(Math.exp(logMin + (logMax - logMin) * i / (n - 1)));
                }
            } else {
                for (let i = 0; i < n; i++) {
                    values.push(min + (max - min) * i / (n - 1));
                }
            }
            return values.map(v => Math.round(v * 1000) / 1000);
        },

        // Parameter options
        parameterOptions: [
            { value: 'pv_capacity_kw', label: 'PV Capacity (kW)', min: 1, max: 20, defaultMin: 2, defaultMax: 8 },
            { value: 'battery_capacity_kwh', label: 'Battery Capacity (kWh)', min: 0, max: 50, defaultMin: 0, defaultMax: 13.5 },
            { value: 'annual_consumption_kwh', label: 'Annual Consumption (kWh)', min: 1000, max: 10000, defaultMin: 2000, defaultMax: 5000 },
            { value: 'n_homes', label: 'Number of Homes', min: 1, max: 500, defaultMin: 10, defaultMax: 100 }
        ],

        get currentParam() {
            return this.parameterOptions.find(p => p.value === this.parameter) || this.parameterOptions[0];
        },

        onParameterChange() {
            const param = this.currentParam;
            this.minVal = param.defaultMin;
            this.maxVal = param.defaultMax;
        },

        // Submit sweep
        async submitSweep() {
            this.submitting = true;
            this.errorMsg = '';
            this.sweepResults = [];

            const payload = {
                parameter: this.parameter,
                min: parseFloat(this.minVal),
                max: parseFloat(this.maxVal),
                steps: parseInt(this.steps),
                mode: this.mode,
                base_config: this.baseConfig,
            };

            try {
                const resp = await fetch('/api/simulate/sweep', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await resp.json();
                if (!resp.ok) {
                    this.errorMsg = data.error || 'Sweep submission failed';
                } else {
                    this.sweepId = data.sweep_id;
                    this.sweepValues = data.values || [];
                    // Populate initial results as pending
                    this.sweepResults = this.sweepValues.map((v, i) => ({
                        value: v,
                        status: 'pending',
                        generation: '-',
                        self_consumption: '-',
                        grid_import: '-',
                        job_id: data.job_ids ? data.job_ids[i] : null,
                    }));
                }
            } catch (e) {
                this.errorMsg = 'Network error: ' + e.message;
            } finally {
                this.submitting = false;
            }
        }
    }));
});
