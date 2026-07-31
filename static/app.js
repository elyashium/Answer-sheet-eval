// Evaluator.ai — Frontend Application
(function () {
    "use strict";

    // DOM Elements
    const uploadZone = document.getElementById("uploadZone");
    const fileInput = document.getElementById("fileInput");
    const uploadContent = document.getElementById("uploadContent");
    const loadingState = document.getElementById("loadingState");
    const loadingStatus = document.getElementById("loadingStatus");
    const fileList = document.getElementById("fileList");
    const fileListItems = document.getElementById("fileListItems");
    const evaluateBtn = document.getElementById("evaluateBtn");
    const resultsSection = document.getElementById("resultsSection");
    const resultsContainer = document.getElementById("resultsContainer");
    const overallScore = document.getElementById("overallScore");

    let selectedFiles = [];
    let lastResults = [];

    // ─── Upload Zone Events ───────────────────────────────────────────

    uploadZone.addEventListener("click", () => fileInput.click());

    uploadZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadZone.classList.add("drag-over");
    });

    uploadZone.addEventListener("dragleave", (e) => {
        e.preventDefault();
        uploadZone.classList.remove("drag-over");
    });

    uploadZone.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadZone.classList.remove("drag-over");
        const files = Array.from(e.dataTransfer.files).filter((f) =>
            f.type.startsWith("image/")
        );
        if (files.length > 0) {
            selectedFiles = files;
            showFileList(files);
        }
    });

    fileInput.addEventListener("change", (e) => {
        const files = Array.from(e.target.files);
        if (files.length > 0) {
            selectedFiles = files;
            showFileList(files);
        }
    });

    // ─── File List Display ────────────────────────────────────────────

    function showFileList(files) {
        fileListItems.innerHTML = "";
        files.forEach((file, index) => {
            const item = document.createElement("div");
            item.className =
                "flex items-center justify-between p-3 bg-white border border-gray-100 rounded-xl";
            item.innerHTML = `
                <div class="flex items-center gap-3">
                    <div class="w-8 h-8 bg-gray-100 rounded-lg flex items-center justify-center">
                        <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                            <rect x="1" y="1" width="12" height="12" rx="2" stroke="#6B7280" stroke-width="1.5"/>
                            <circle cx="4.5" cy="5" r="1.5" fill="#6B7280"/>
                            <path d="M1 10L4 7L6 9L9 5L13 10" stroke="#6B7280" stroke-width="1.5" stroke-linecap="round"/>
                        </svg>
                    </div>
                    <div>
                        <p class="text-sm font-medium text-gray-700">${file.name}</p>
                        <p class="text-xs text-gray-400">${(file.size / 1024 / 1024).toFixed(2)} MB</p>
                    </div>
                </div>
                <button onclick="removeFile(${index})" class="text-gray-300 hover:text-red-500 transition-colors">
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                        <path d="M4 4L12 12M12 4L4 12" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
                    </svg>
                </button>
            `;
            fileListItems.appendChild(item);
        });
        fileList.classList.remove("hidden");
    }

    // Expose to global for inline onclick
    window.removeFile = function (index) {
        selectedFiles.splice(index, 1);
        if (selectedFiles.length === 0) {
            fileList.classList.add("hidden");
        } else {
            showFileList(selectedFiles);
        }
    };

    // ─── Submit & Evaluate ────────────────────────────────────────────

    window.submitFiles = async function () {
        if (selectedFiles.length === 0) return;

        // Show loading state
        uploadContent.classList.add("hidden");
        loadingState.classList.remove("hidden");
        fileList.classList.add("hidden");
        resultsSection.classList.add("hidden");

        const statusMessages = [
            "Extracting handwritten text...",
            "Running semantic analysis...",
            "Consulting LLM judge...",
            "Calibrating confidence scores...",
        ];

        let msgIndex = 0;
        const statusInterval = setInterval(() => {
            msgIndex = (msgIndex + 1) % statusMessages.length;
            loadingStatus.textContent = statusMessages[msgIndex];
        }, 2500);

        try {
            const formData = new FormData();
            selectedFiles.forEach((file) => formData.append("files", file));

            const response = await fetch("/api/evaluate", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || "Evaluation failed");
            }

            const data = await response.json();
            lastResults = data.results;
            renderResults(data.results);
        } catch (error) {
            showError(error.message);
        } finally {
            clearInterval(statusInterval);
            loadingState.classList.add("hidden");
            uploadContent.classList.remove("hidden");
        }
    };

    // ─── Render Results ───────────────────────────────────────────────

    function renderResults(results) {
        resultsContainer.innerHTML = "";
        resultsSection.classList.remove("hidden");

        results.forEach((fileResult) => {
            // Overall score header
            overallScore.innerHTML = `
                <div class="text-right">
                    <p class="text-sm text-gray-400 mb-1">${fileResult.subject} — Total</p>
                    <p class="text-2xl font-display font-semibold">${fileResult.total_scored} / ${fileResult.total_marks}</p>
                    <p class="text-sm text-gray-400">${fileResult.overall_percentage}%</p>
                </div>
            `;

            // Question result cards
            fileResult.questions.forEach((q, index) => {
                const card = document.createElement("div");
                card.className = "result-card fade-in";
                card.style.animationDelay = `${index * 0.1}s`;

                const confidenceClass = `confidence-${q.confidence?.confidence_color || "red"}`;
                const confidenceLabel = q.confidence?.confidence_label || "unknown";
                const percentage = q.percentage || 0;
                const progressColor =
                    percentage >= 70 ? "#16A34A" : percentage >= 40 ? "#CA8A04" : "#DC2626";

                card.innerHTML = `
                    <div class="flex items-start justify-between mb-4">
                        <div>
                            <div class="flex items-center gap-3 mb-1">
                                <h3 class="text-lg font-semibold">${q.question_id}</h3>
                                <span class="confidence-badge ${confidenceClass}">
                                    <span class="w-2 h-2 rounded-full bg-current"></span>
                                    ${confidenceLabel}
                                </span>
                                ${q.confidence?.needs_human_review ? '<span class="text-xs text-orange-600 bg-orange-50 px-2 py-1 rounded-full">⚠ Review</span>' : ""}
                            </div>
                            <p class="text-sm text-gray-500">${q.question || ""}</p>
                        </div>
                        <div class="score-ring" style="border-color: ${progressColor}">
                            <span style="color: ${progressColor}">${q.marks_awarded}/${q.max_marks}</span>
                        </div>
                    </div>

                    <!-- Progress Bar -->
                    <div class="progress-bar mb-5">
                        <div class="progress-fill" style="width: ${percentage}%; background: ${progressColor}"></div>
                    </div>

                    <!-- Extracted Text -->
                    <div class="mb-4">
                        <p class="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Extracted Answer</p>
                        <p class="text-sm text-gray-700 bg-gray-50 rounded-xl p-4 leading-relaxed">${escapeHtml(q.extracted_text || "")}</p>
                    </div>

                    <!-- LLM Reasoning -->
                    ${q.reasoning ? `
                    <div class="mb-4">
                        <p class="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">AI Reasoning</p>
                        <p class="text-sm text-gray-600 leading-relaxed">${escapeHtml(q.reasoning)}</p>
                    </div>
                    ` : ""}

                    <!-- Strengths & Missing Concepts -->
                    <div class="grid grid-cols-2 gap-4">
                        ${q.strengths && q.strengths.length > 0 ? `
                        <div>
                            <p class="text-xs font-semibold text-green-700 uppercase tracking-wider mb-2">Strengths</p>
                            <ul class="space-y-1">
                                ${q.strengths.map((s) => `<li class="text-xs text-gray-600 flex items-start gap-1"><span class="text-green-500 mt-0.5">✓</span> ${escapeHtml(s)}</li>`).join("")}
                            </ul>
                        </div>
                        ` : ""}
                        ${q.missing_concepts && q.missing_concepts.length > 0 ? `
                        <div>
                            <p class="text-xs font-semibold text-red-700 uppercase tracking-wider mb-2">Missing Concepts</p>
                            <ul class="space-y-1">
                                ${q.missing_concepts.map((c) => `<li class="text-xs text-gray-600 flex items-start gap-1"><span class="text-red-400 mt-0.5">✗</span> ${escapeHtml(c)}</li>`).join("")}
                            </ul>
                        </div>
                        ` : ""}
                    </div>

                    <!-- Scoring Breakdown -->
                    <div class="mt-5 pt-4 border-t border-gray-100">
                        <div class="flex gap-6 text-xs text-gray-400">
                            <span>Semantic: <strong class="text-gray-600">${(q.scoring?.semantic_similarity * 100 || 0).toFixed(0)}%</strong></span>
                            <span>LLM Judge: <strong class="text-gray-600">${(q.scoring?.llm_score * 100 || 0).toFixed(0)}%</strong></span>
                            <span>Concepts: <strong class="text-gray-600">${(q.scoring?.concepts_coverage * 100 || 0).toFixed(0)}%</strong></span>
                            <span>Confidence: <strong class="text-gray-600">${(q.confidence?.confidence_score * 100 || 0).toFixed(0)}%</strong></span>
                        </div>
                    </div>
                `;

                resultsContainer.appendChild(card);
            });
        });

        // Scroll to results
        setTimeout(() => {
            resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
            const exportAction = document.getElementById("exportAction");
            if (exportAction) exportAction.classList.remove("hidden");
        }, 200);
    }
    
    // ─── Export CSV ───────────────────────────────────────────────────

    window.downloadCSV = async function () {
        if (!lastResults || lastResults.length === 0) return;
        
        try {
            const response = await fetch("/api/export/csv", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ results: lastResults })
            });
            
            if (!response.ok) throw new Error("Failed to generate CSV");
            
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "evaluation_results.csv";
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            a.remove();
        } catch (error) {
            console.error("CSV Export failed:", error);
            alert("Failed to export CSV. Please check the console.");
        }
    };

    // ─── Error Display ────────────────────────────────────────────────

    function showError(message) {
        resultsContainer.innerHTML = `
            <div class="result-card border-red-200 fade-in">
                <div class="flex items-center gap-3 mb-2">
                    <div class="w-8 h-8 bg-red-100 rounded-full flex items-center justify-center">
                        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                            <path d="M8 4V8M8 11H8.01" stroke="#DC2626" stroke-width="2" stroke-linecap="round"/>
                            <circle cx="8" cy="8" r="6" stroke="#DC2626" stroke-width="1.5"/>
                        </svg>
                    </div>
                    <h3 class="text-lg font-semibold text-red-700">Evaluation Failed</h3>
                </div>
                <p class="text-sm text-red-600">${escapeHtml(message)}</p>
            </div>
        `;
        resultsSection.classList.remove("hidden");
    }

    // ─── Utility ──────────────────────────────────────────────────────

    function escapeHtml(text) {
        const map = {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#039;",
        };
        return text.replace(/[&<>"']/g, (m) => map[m]);
    }
})();