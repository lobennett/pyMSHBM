"""Discover hemisphere-paired fsaverage6 BOLD in fMRIPrep derivatives.

This is a deliberately small filename reader, not a general BIDS validator.
Discovery does not open imaging payloads, denoise data, or resample meshes.
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


class BIDSInputError(ValueError):
    """The selected input cannot be interpreted as unambiguous surface runs."""


@dataclass(frozen=True)
class SurfaceRun:
    """One acquired BOLD run, used as one within-subject model session.

    Labels omit their BIDS prefix. ``entities`` preserves all filename entities
    except hemisphere as immutable key/value pairs for output provenance.
    """

    subject: str
    session: str | None
    task: str
    run: str | None
    lh: Path
    rh: Path
    entities: tuple[tuple[str, str], ...] = ()


def _labels(values: Sequence[str] | str | None, prefix: str) -> set[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        values = [values]
    result = {value.removeprefix(prefix + "-") for value in values}
    if not result or any(not value.isascii() or not value.isalnum() for value in result):
        raise BIDSInputError(f"Provide nonempty alphanumeric {prefix} labels.")
    return result


def _parse_entities(path: Path, extension: str) -> dict[str, str]:
    entities = {}
    tokens = path.name.removesuffix(extension).split("_")
    for token in tokens[:-1]:
        key, sep, value = token.partition("-")
        if (not sep or not key.isascii() or not key.isalnum() or
                not value.isascii() or not value.isalnum() or key in entities):
            raise BIDSInputError(f"Malformed or duplicate BIDS entity in {path.name}.")
        entities[key] = value
    if "sub" not in entities or "task" not in entities:
        raise BIDSInputError(f"BOLD filename requires sub and task entities: {path}")
    return entities


def discover_surface_runs(
    input_dir: str | Path,
    *,
    participant_labels: Sequence[str] | str | None = None,
    session_labels: Sequence[str] | str | None = None,
    task: str = "rest",
    run_labels: Sequence[str] | str | None = None,
) -> list[SurfaceRun]:
    """Find fsaverage6 GIFTI pairs in a derivative root or BIDS dataset.

    Also accepts ``space-fsaverage_den-41k``. All entities must agree across
    hemispheres. Alternative spaces, densities, descriptions, echoes, or
    reconstructions of one acquisition are ambiguous: provide a derivative
    directory containing exactly one chosen representation per physical run.
    Acquisition and phase-encoding direction distinguish separate runs.

    This validates names and pairing only; the workflow checks actual mesh
    dimensions, timepoints and finite values before fitting.
    """
    supplied = Path(input_dir).expanduser().resolve()
    if not supplied.is_dir():
        raise BIDSInputError(f"Input directory does not exist: {supplied}")
    nested = supplied / "derivatives" / "fmriprep"
    root = nested if nested.is_dir() else supplied
    # fMRIPrep's pre-21 legacy layout has an extra fmriprep directory.
    if root == supplied and (root / "fmriprep").is_dir():
        root /= "fmriprep"
    participants = _labels(participant_labels, "sub")
    sessions = _labels(session_labels, "ses")
    selected_runs = _labels(run_labels, "run")
    task = task.removeprefix("task-")
    if not task.isascii() or not task.isalnum():
        raise BIDSInputError("Provide a nonempty alphanumeric task label.")

    groups: dict[tuple[tuple[str, str], ...], dict[str, Path]] = defaultdict(dict)
    unsupported = []
    cifti = []
    volumes = []
    folders = sorted([*root.glob("sub-*/func"), *root.glob("sub-*/ses-*/func")])
    for folder in folders:
        for path in sorted(folder.iterdir()):
            if not path.is_file():
                continue
            extension = next((ext for ext in (".func.gii", ".dtseries.nii", ".nii.gz", ".nii")
                              if path.name.endswith("_bold" + ext)), None)
            if extension is None:
                continue
            entities = _parse_entities(path, extension)
            if entities["task"] != task:
                continue
            if participants is not None and entities["sub"] not in participants:
                continue
            if sessions is not None and entities.get("ses") not in sessions:
                continue
            if selected_runs is not None and entities.get("run") not in selected_runs:
                continue
            directory_session = folder.parent.name if folder.parent.name.startswith("ses-") else None
            directory_subject = folder.parent.parent.name if directory_session else folder.parent.name
            if (directory_subject != "sub-" + entities["sub"] or
                    directory_session != ("ses-" + entities["ses"] if "ses" in entities else None)):
                raise BIDSInputError(f"Filename subject/session disagrees with its directory: {path}")
            if extension == ".dtseries.nii":
                cifti.append(path)
                continue
            if extension != ".func.gii":
                volumes.append(path)
                continue
            space, den = entities.get("space"), entities.get("den")
            supported = ((space == "fsaverage6" and den in (None, "41k")) or
                         (space == "fsaverage" and den == "41k"))
            if not supported:
                unsupported.append(path)
                continue
            hemi = entities.pop("hemi", None)
            if hemi not in ("L", "R"):
                raise BIDSInputError(f"Missing or invalid hemisphere (use hemi-L or hemi-R): {path}")
            key = tuple(sorted(entities.items()))
            if hemi in groups[key]:
                raise BIDSInputError(f"Duplicate hemisphere {hemi} for one run: {groups[key][hemi]} and {path}")
            groups[key][hemi] = path

    if not groups:
        guidance = "Provide paired hemi-L/hemi-R BOLD GIFTIs in fsaverage6 (or space-fsaverage_den-41k)."
        if cifti:
            raise BIDSInputError(f"CIFTI input is incompatible with this workflow. {guidance} Export these surfaces from fMRIPrep; no automatic resampling is performed.")
        if unsupported:
            raise BIDSInputError(f"Unsupported surface mesh or density: {unsupported[0].name}. {guidance}")
        if volumes:
            raise BIDSInputError(f"Only volumetric BOLD found; raw BIDS cannot be fitted directly. Run fMRIPrep with --output-spaces fsaverage6 and supply its derivatives. {guidance}")
        filters = f"task={task}, participants={sorted(participants) if participants else 'all'}, sessions={sorted(sessions) if sessions else 'all'}, runs={sorted(selected_runs) if selected_runs else 'all'}"
        raise BIDSInputError(f"No matching surface BOLD files in {root} ({filters}). {guidance}")

    runs = []
    acquisitions = {}
    for key, hemis in sorted(groups.items()):
        if set(hemis) != {"L", "R"}:
            missing = "R" if "L" in hemis else "L"
            raise BIDSInputError(f"Missing hemisphere {missing} for {next(iter(hemis.values()))}. Both hemispheres must match all BIDS entities, including space, den and desc.")
        entities = dict(key)
        physical = tuple((k, entities[k]) for k in ("sub", "ses", "task", "run", "acq", "dir") if k in entities)
        if physical in acquisitions:
            raise BIDSInputError(f"Ambiguous variants of one physical run: {acquisitions[physical].name} and {hemis['L'].name}. Supply one chosen echo/reconstruction/description/space per acquisition in the input derivative directory.")
        acquisitions[physical] = hemis["L"]
        runs.append(SurfaceRun(entities["sub"], entities.get("ses"), entities["task"],
                               entities.get("run"), hemis["L"], hemis["R"], key))
    for requested, key, label in ((participants, "sub", "participant"),
                                  (sessions, "ses", "session"),
                                  (selected_runs, "run", "run")):
        if requested is not None:
            found = {dict(record.entities).get(key) for record in runs}
            missing = requested - found
            if missing:
                raise BIDSInputError(f"No matching surface BOLD for requested {label} labels: {', '.join(sorted(missing))}.")
    return sorted(runs, key=lambda record: (record.subject, record.session or "", record.task,
                                           record.run or "", record.entities))
