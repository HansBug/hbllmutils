.PHONY: docs test unittest tbuild build

PYTHON ?= $(shell which python)

PROJ_DIR := ${CURDIR}
PROJ_NAME := hbllmutils

DOC_DIR    := ${PROJ_DIR}/docs
TEST_DIR   := ${PROJ_DIR}/test
SRC_DIR    := ${PROJ_DIR}/${PROJ_NAME}
DIST_DIR   := ${PROJ_DIR}/dist
TBUILD_DIR := ${PROJ_DIR}/tbuild
BUILD_DIR  := ${PROJ_DIR}/build

RANGE_DIR          ?= .
RANGE_TEST_DIR     := ${TEST_DIR}/${RANGE_DIR}
RANGE_SRC_DIR      := ${SRC_DIR}/${RANGE_DIR}
RANGE_SRC_DIR_TEST ?= $(shell python -m tools.make_test_file -i "${RANGE_SRC_DIR}" -s "${SRC_DIR}" -t "${TEST_DIR}")

#PYTHON_CODE_DIR   := ${SRC_DIR}/${RANGE_DIR}
#RST_DOC_DIR       := ${DOC_DIR}/source/api_doc/${RANGE_DIR}
PYTHON_CODE_DIR   := ${SRC_DIR}
RST_DOC_DIR       := ${DOC_DIR}/source/api_doc
PYTHON_CODE_FILES := $(shell find ${PYTHON_CODE_DIR} -name "*.py" ! -name "__*.py" 2>/dev/null)
RST_DOC_FILES     := $(patsubst ${PYTHON_CODE_DIR}/%.py,${RST_DOC_DIR}/%.rst,${PYTHON_CODE_FILES})
PYTHON_NONM_FILES := $(shell find ${PYTHON_CODE_DIR} -name "__init__.py" 2>/dev/null)
RST_NONM_FILES    := $(foreach file,${PYTHON_NONM_FILES},$(patsubst %/__init__.py,%/index.rst,$(patsubst ${PYTHON_CODE_DIR}/%,${RST_DOC_DIR}/%,$(patsubst ${PYTHON_CODE_DIR}/__init__.py,${RST_DOC_DIR}/index.rst,${file}))))

AUTO_OPTIONS ?= --param max_tokens=400000 --param temperature=0.5 --no-ignore-module hbutils --model-name gpt-5.2-codex

COV_TYPES ?= xml term-missing

package:
	$(PYTHON) -m build --sdist --wheel --outdir ${DIST_DIR}
build:
	pyinstaller -D -F $(shell python -m tools.resources) -n ${PROJ_NAME} -c ${PROJ_NAME}_cli.py
clean:
	rm -rf ${DIST_DIR} ${BUILD_DIR} *.egg-info
	rm -rf build dist ${PROJ_NAME}.spec

test: unittest

unittest:
	pytest "${RANGE_TEST_DIR}" \
		-sv -m unittest \
		$(shell for type in ${COV_TYPES}; do echo "--cov-report=$$type"; done) \
		--cov="${RANGE_SRC_DIR}" \
		$(if ${MIN_COVERAGE},--cov-fail-under=${MIN_COVERAGE},) \
		$(if ${WORKERS},-n ${WORKERS},)

docs:
	$(MAKE) -C "${DOC_DIR}" build
pdocs:
	$(MAKE) -C "${DOC_DIR}" prod
docs_auto:
	python -m ${PROJ_NAME} code pydoc -i "${RANGE_SRC_DIR}" ${AUTO_OPTIONS}
todos_auto:
	python -m ${PROJ_NAME} code todo -i "${RANGE_SRC_DIR}" ${AUTO_OPTIONS}
tests_auto:
	python -m ${PROJ_NAME} code unittest -i "${RANGE_SRC_DIR}" -o "${RANGE_SRC_DIR_TEST}" \
		${AUTO_OPTIONS}
rst_auto: ${RST_DOC_FILES} ${RST_NONM_FILES} auto_rst_top_index.py
	python auto_rst_top_index.py -i ${PYTHON_CODE_DIR} -o ${DOC_DIR}/source/api_doc.rst
${RST_DOC_DIR}/%.rst: ${PYTHON_CODE_DIR}/%.py auto_rst.py Makefile
	@mkdir -p $(dir $@)
	python auto_rst.py -i $< -o $@
${RST_DOC_DIR}/%/index.rst: ${PYTHON_CODE_DIR}/%/__init__.py auto_rst.py Makefile
	@mkdir -p $(dir $@)
	python auto_rst.py -i $< -o $@
${RST_DOC_DIR}/index.rst: ${PYTHON_CODE_DIR}/__init__.py auto_rst.py Makefile
	@mkdir -p $(dir $@)
	python auto_rst.py -i $< -o $@

update_pypi_downloads:
	python -m tools.pypi_downloads --dst-csv-file "${RANGE_SRC_DIR}/meta/code/pypi_downloads.csv"
