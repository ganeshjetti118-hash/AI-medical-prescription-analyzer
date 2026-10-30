# Medicine name database

`medicine_names.csv` holds the names used to verify medicines extracted from a
prescription. It has three columns: `name`, `type` (`generic` | `brand`), `source`.

## Provenance and licence

- **Source:** FDA Orange Book data files (`products.txt`).
  <https://www.fda.gov/drug/drugs-fda-orange-book>
- **Licence:** U.S. Government work — public domain, free to use and redistribute.
- **Retrieved:** 2026-10-30 (8,229 unique names: 3,050 generic, 5,179 brand).
- **No patient data** is included; these are drug product names only.

## Coverage and limitations

- Coverage is US-approved products: generic (USAN/INN) ingredient names plus US
  brand names. It does **not** contain Indian brand names (e.g. Dolo, Crocin),
  which are not freely licensed. Names outside this list are reported as
  "no database match" and must be confirmed by the user.
- The database is a spelling aid, **not** clinical validation. A match does not
  confirm the drug, dose, or appropriateness of a prescription.

## Refresh

```bash
docker compose -f docker-compose.base44.yml exec -T web python scripts/build_medicine_db.py
```

The script re-downloads the Orange Book, re-parses it, and rewrites
`medicine_names.csv`. Any manually added rows are replaced, so keep custom
entries in a separate file and pass it with `MedicineVerifier(csv_path=...)`.

## INN ↔ USAN synonym map

`inn_aliases.csv` maps International Nonproprietary Names used in India and
other markets to the US name held in the Orange Book (e.g. `paracetamol` →
`acetaminophen`, `salbutamol` → `albuterol`, `frusemide` → `furosemide`).

- **Source:** curated by hand from WHO INN listings and standard pharmacopoeial
  naming; 27 pairs covering the most common prescription generics.
- **Behaviour:** an alias hit is shown as an *equivalent name* (98% score), not
  as an exact string match, and still asks the user to confirm.
- Extend the file with more `inn,usan` rows; rows whose US name is absent from
  `medicine_names.csv` are ignored at load time.

## Adding your own licensed list

Use the same `name,type,source` columns. To fold your list into the bundled
database, append rows and set `source` accordingly — keep the licence note here
up to date.
