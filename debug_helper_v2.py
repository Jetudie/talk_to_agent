import os
import re
import warnings
import configparser
import numpy as np
import pandas as pd
from bitstring import BitStream
from elftools.elf.elffile import ELFFile
from elftools.dwarf.dwarf_expr import DWARFExprParser

__all__ = [
    'MemoryMap',
    'DwarfIndex',
    'ExpressionEvaluator',
    'TableParser',
    'MemoryDumper',
    'RegisterParser',
    'QueryEngine',
    'format_text_table',
]

# ==============================================================================
# 0. File Stitching (Global Memory Map)
# ==============================================================================
class MemoryMap:
    """
    Parses and stores memory dumps from binary files.
    Filenames must follow the pattern: {RAM_NAME}_{START_ADDRESS}_{LENGTH_BYTE}.bin
    START_ADDRESS and LENGTH_BYTE can be hex (0x prefix) or decimal.
    Reading memory outside dumped ranges returns zeros (zero-fill on read).
    """
    def __init__(self):
        # Format: { start_addr: bytearray }
        self.segments = {}

    def load_bin_files(self, directory):
        """Find and load all matching .bin files in directory into the memory map."""
        pattern = re.compile(
            r'^(?P<name>.*)_(?P<start>(?:0x)?[0-9a-fA-F]+)_(?P<length>(?:0x)?[0-9a-fA-F]+)\.bin$',
            re.IGNORECASE
        )
        
        loaded_count = 0
        for filename in sorted(os.listdir(directory)):
            match = pattern.match(filename)
            if match:
                start_addr = int(match.group('start'), 0)
                length = int(match.group('length'), 0)
                filepath = os.path.join(directory, filename)
                
                with open(filepath, 'rb') as f:
                    data = f.read()
                    if len(data) != length:
                        print(f"[Warning] {filename} size ({len(data)} B) differs from filename length ({length} B)")
                    self.segments[start_addr] = bytearray(data)
                    loaded_count += 1
                print(f"[Loaded] {filename} @ {hex(start_addr)} ({len(data)} bytes)")
        return loaded_count

    def add_segment(self, start_addr, data):
        """Manually add a memory segment (useful for testing or direct byte additions)."""
        self.segments[start_addr] = bytearray(data)

    def read_memory(self, target_addr, size):
        """
        Read `size` bytes from `target_addr`.
        Zero-fills any addresses that are not present in loaded segments.
        """
        result = bytearray(size)
        target_end = target_addr + size
        
        for seg_addr, seg_data in self.segments.items():
            seg_end = seg_addr + len(seg_data)
            
            overlap_start = max(target_addr, seg_addr)
            overlap_end = min(target_end, seg_end)
            
            if overlap_start < overlap_end:
                res_offset = overlap_start - target_addr
                seg_offset = overlap_start - seg_addr
                copy_len = overlap_end - overlap_start
                result[res_offset : res_offset + copy_len] = seg_data[seg_offset : seg_offset + copy_len]
                
        return bytes(result)

    def write_memory(self, target_addr, data):
        """Write bytes to memory map, updating existing segments in-place when overlapping."""
        data = bytearray(data)
        write_end = target_addr + len(data)

        # Try to write into existing segments that overlap
        remaining = bytearray(len(data))  # tracks which bytes have been placed
        placed = [False] * len(data)

        for seg_addr, seg_data in self.segments.items():
            seg_end = seg_addr + len(seg_data)
            overlap_start = max(target_addr, seg_addr)
            overlap_end = min(write_end, seg_end)
            if overlap_start < overlap_end:
                seg_offset = overlap_start - seg_addr
                data_offset = overlap_start - target_addr
                copy_len = overlap_end - overlap_start
                seg_data[seg_offset : seg_offset + copy_len] = data[data_offset : data_offset + copy_len]
                for k in range(data_offset, data_offset + copy_len):
                    placed[k] = True

        # If any bytes were not placed into existing segments, create a new segment
        if not all(placed):
            self.segments[target_addr] = bytearray(data)


# ==============================================================================
# 1. DWARF & C Expression Evaluation (.axf / ELF integration)
# ==============================================================================
def unwrap_type(die):
    """Recursively unwrap typedef, const, volatile DWARF tags to get core type DIE."""
    curr = die
    while curr is not None:
        if curr.tag in ('DW_TAG_typedef', 'DW_TAG_const_type', 'DW_TAG_volatile_type'):
            if 'DW_AT_type' in curr.attributes:
                curr = curr.get_DIE_from_attribute('DW_AT_type')
            else:
                break
        else:
            break
    return curr

def get_member_offset(die):
    """Extract byte offset from DW_TAG_member."""
    attr = die.attributes.get('DW_AT_data_member_location')
    if attr is None:
        return 0
    val = attr.value
    if isinstance(val, int):
        return val
    parser = DWARFExprParser(die.cu.structs)
    ops = parser.parse_expr(val)
    for op in ops:
        if op.op_name == 'DW_OP_plus_uconst':
            return op.args[0]
    return 0

def get_variable_address(die):
    """Extract address from global DW_TAG_variable."""
    if 'DW_AT_specification' in die.attributes:
        spec_die = die.get_DIE_from_attribute('DW_AT_specification')
        addr = get_variable_address(spec_die)
        if addr is not None:
            return addr

    attr = die.attributes.get('DW_AT_location')
    if attr is None:
        return None
    val = attr.value
    parser = DWARFExprParser(die.cu.structs)
    ops = parser.parse_expr(val)
    for op in ops:
        if op.op_name == 'DW_OP_addr':
            return op.args[0]
    return None

def compute_type_size(die):
    """Compute size in bytes of DWARF type DIE."""
    die = unwrap_type(die)
    if die is None:
        return 0

    if 'DW_AT_byte_size' in die.attributes:
        return die.attributes['DW_AT_byte_size'].value

    if die.tag == 'DW_TAG_pointer_type':
        return die.cu.structs.initial_length_size

    if die.tag == 'DW_TAG_array_type':
        elem_die = die.get_DIE_from_attribute('DW_AT_type') if 'DW_AT_type' in die.attributes else None
        elem_size = compute_type_size(elem_die) if elem_die else 0
        total_count = 1
        for child in die.iter_children():
            if child.tag == 'DW_TAG_subrange_type':
                if 'DW_AT_count' in child.attributes:
                    total_count *= child.attributes['DW_AT_count'].value
                elif 'DW_AT_upper_bound' in child.attributes:
                    total_count *= (child.attributes['DW_AT_upper_bound'].value + 1)
        return elem_size * total_count

    return 0

class DwarfIndex:
    """Indexes DWARF info from an AXF/ELF file for quick symbol and type lookups."""
    def __init__(self, axf_path):
        self.axf_path = axf_path
        self.global_vars = {}
        self.type_dies = {}
        self._load_and_index()

    def _load_and_index(self):
        with open(self.axf_path, 'rb') as f:
            elffile = ELFFile(f)
            if not elffile.has_dwarf_info():
                raise ValueError(f"File {self.axf_path} has no DWARF debug information.")
            
            dwarfinfo = elffile.get_dwarf_info()
            for cu in dwarfinfo.iter_CUs():
                for die in cu.iter_DIEs():
                    name_attr = die.attributes.get('DW_AT_name')
                    if not name_attr:
                        continue
                    name = name_attr.value.decode('utf-8', errors='ignore')

                    if die.tag == 'DW_TAG_variable':
                        addr = get_variable_address(die)
                        type_die = die.get_DIE_from_attribute('DW_AT_type') if 'DW_AT_type' in die.attributes else None
                        if addr is not None:
                            self.global_vars[name] = {
                                'address': addr,
                                'type_die': type_die,
                                'die': die
                            }
                    elif die.tag in ('DW_TAG_structure_type', 'DW_TAG_union_type', 
                                    'DW_TAG_typedef', 'DW_TAG_enumeration_type'):
                        if name not in self.type_dies:
                            self.type_dies[name] = die

class ExpressionEvaluator:
    """
    Evaluates C expressions such as:
    - Global variables: `gdwA.field_B.field_C[0]`
    - Explicit casts: `((struct_name)(0xf8f81100)).field_D[10]`
    Calculates final address, size, and reads value from MemoryMap.
    """
    def __init__(self, memory_map=None, dwarf_index=None, mem_map=None):
        if mem_map is not None:
            warnings.warn(
                "The 'mem_map' parameter is deprecated; use 'memory_map' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.memory_map = memory_map if memory_map is not None else mem_map
        self.dwarf_index = dwarf_index

    def evaluate(self, expression):
        """
        Returns a dict:
        {
          'expression': str,
          'address': int,
          'size': int,
          'raw_bytes': bytes,
          'value_int': int,
          'value_hex': str
        }
        """
        expr_str = expression.strip()

        if self.dwarf_index is None:
            # Fallback simple regex evaluation if no DWARF index present
            return self._fallback_eval(expr_str)

        # 1. Check cast syntax: ((type_name)(address)).field_path
        cast_match = re.match(r'^\(\(\s*([a-zA-Z0-9_]+)\s*\)\(\s*(0x[0-9a-fA-F]+|[0-9]+)\s*\)\)(.*)', expr_str)
        if cast_match:
            type_name = cast_match.group(1)
            base_addr = int(cast_match.group(2), 0)
            access_path = cast_match.group(3)

            if type_name not in self.dwarf_index.type_dies:
                raise ValueError(f"Type '{type_name}' not found in DWARF symbols.")
            root_die = self.dwarf_index.type_dies[type_name]
            addr, size, _ = self._walk_path(base_addr, root_die, access_path)
            return self._build_result(expr_str, addr, size)

        # 2. Check global var syntax: gdwA.field_B.field_C[0]
        var_match = re.match(r'^([a-zA-Z0-9_]+)(.*)', expr_str)
        if var_match:
            var_name = var_match.group(1)
            access_path = var_match.group(2)

            if var_name not in self.dwarf_index.global_vars:
                raise ValueError(f"Global variable '{var_name}' not found in DWARF symbols.")

            var_info = self.dwarf_index.global_vars[var_name]
            base_addr = var_info['address']
            root_die = var_info['type_die']
            addr, size, _ = self._walk_path(base_addr, root_die, access_path)
            return self._build_result(expr_str, addr, size)

        raise ValueError(f"Unable to parse expression format: {expr_str}")

    def _walk_path(self, current_addr, current_die, path_str):
        tokens = re.findall(r'\.([a-zA-Z0-9_]+)|\[([0-9]+)\]', path_str)

        for member_name, array_idx in tokens:
            current_die = unwrap_type(current_die)
            if current_die is None:
                raise ValueError("Target type resolved to None during path traversal.")

            if member_name:
                if current_die.tag not in ('DW_TAG_structure_type', 'DW_TAG_union_type'):
                    raise ValueError(f"Cannot access member '{member_name}' on non-struct/union type DIE '{current_die.tag}'")

                found = False
                for child in current_die.iter_children():
                    if child.tag == 'DW_TAG_member':
                        name_attr = child.attributes.get('DW_AT_name')
                        if name_attr and name_attr.value.decode('utf-8') == member_name:
                            offset = get_member_offset(child) if current_die.tag == 'DW_TAG_structure_type' else 0
                            current_addr += offset
                            current_die = child.get_DIE_from_attribute('DW_AT_type') if 'DW_AT_type' in child.attributes else None
                            found = True
                            break

                if not found:
                    raise ValueError(f"Member '{member_name}' not found in type DIE")

            elif array_idx:
                idx = int(array_idx)
                if current_die.tag == 'DW_TAG_array_type':
                    elem_die = current_die.get_DIE_from_attribute('DW_AT_type') if 'DW_AT_type' in current_die.attributes else None
                    elem_size = compute_type_size(elem_die) if elem_die else 0
                    current_addr += idx * elem_size
                    current_die = elem_die
                elif current_die.tag == 'DW_TAG_pointer_type':
                    if self.memory_map:
                        ptr_bytes = self.memory_map.read_memory(current_addr, current_die.cu.structs.initial_length_size)
                        current_addr = int.from_bytes(ptr_bytes, byteorder='little')
                    target_die = current_die.get_DIE_from_attribute('DW_AT_type') if 'DW_AT_type' in current_die.attributes else None
                    elem_size = compute_type_size(target_die) if target_die else 0
                    current_addr += idx * elem_size
                    current_die = target_die
                else:
                    raise ValueError(f"Cannot index non-array/non-pointer type DIE '{current_die.tag}'")

        final_die = unwrap_type(current_die)
        final_size = compute_type_size(final_die)
        if final_size == 0:
            final_size = 4 # default to 4 bytes fallback if un-sized
        return current_addr, final_size, final_die

    def _fallback_eval(self, expr_str):
        # Basic fallback for demo/testing without full DWARF index
        # Trailing .field_path is optional so bare casts like ((type)(0xaddr)) work too
        cast_match = re.search(r'\(\(.*?\)\((0x[0-9a-fA-F]+|[0-9]+)\)\)(?:\.(.*?))?$', expr_str)
        if cast_match:
            base_addr = int(cast_match.group(1), 0)
            return self._build_result(expr_str, base_addr, 4)
        
        # Plain hex address fallback if passed directly
        if expr_str.startswith('0x'):
            addr = int(expr_str, 0)
            return self._build_result(expr_str, addr, 4)

        raise ValueError("No DWARF index loaded and expression is not a simple address cast.")

    def _build_result(self, expr_str, address, size):
        raw_bytes = self.memory_map.read_memory(address, size)
        val_int = int.from_bytes(raw_bytes, byteorder='little')
        val_hex = f"0x{val_int:0{size*2}x}"
        return {
            'expression': expr_str,
            'address': address,
            'size': size,
            'raw_bytes': raw_bytes,
            'value_int': val_int,
            'value_hex': val_hex
        }


# ==============================================================================
# 2. RAM Structured Slice Table & Output Formats
# ==============================================================================
class TableParser:
    """
    Slices RAM into fixed-length entries according to user layout.
    Checks field bit lengths sum and entry alignment.
    Generates pandas DataFrame with formatted text output or CSV.
    """
    def __init__(self, memory_map):
        self.memory_map = memory_map

    def parse_memory_to_table(self, start_addr, total_bytes, layout, endian='big'):
        """
        `layout` can be:
        - a pandas DataFrame
        - a list of dicts: [{'name': 'F1', 'bits': 8, 'format': 'hex'}, ...]

        `endian`: 'big' (default, raw bit-stream order) or 'little'.
                  When 'little', multi-byte fields have their bytes reversed before interpretation.

        Validates total bit length against byte alignment.
        Returns pandas DataFrame containing both display values and internal numeric values (`_{col}_int`).
        """
        if isinstance(layout, list):
            layout_df = pd.DataFrame(layout)
        else:
            layout_df = layout.copy()

        if 'bits' not in layout_df.columns or 'name' not in layout_df.columns:
            raise ValueError("Layout must contain 'name' and 'bits' columns.")

        total_bits = layout_df['bits'].sum()
        if total_bits % 8 != 0:
            raise ValueError(f"Error: Field bit length sum ({total_bits} bits) is not byte-aligned (must be multiple of 8).")

        entry_bytes = total_bits // 8
        if total_bytes % entry_bytes != 0:
            raise ValueError(f"Error: total_bytes ({total_bytes}) is not a multiple of entry size ({entry_bytes} bytes / {total_bits} bits).")

        num_entries = total_bytes // entry_bytes
        raw_data = self.memory_map.read_memory(start_addr, total_bytes)
        stream = BitStream(raw_data)

        parsed_entries = []
        for i in range(num_entries):
            entry_data = {'entry_idx': i, 'entry_addr': f"0x{start_addr + i * entry_bytes:08x}"}
            for _, row in layout_df.iterrows():
                name = row['name']
                bits = int(row['bits'])
                fmt = str(row.get('format', 'hex')).lower()

                val = stream.read(f'uint:{bits}')

                # Byte-swap for little-endian interpretation of multi-byte fields
                if endian == 'little' and bits >= 16 and bits % 8 == 0:
                    byte_count = bits // 8
                    val_bytes = val.to_bytes(byte_count, byteorder='big')
                    val = int.from_bytes(val_bytes, byteorder='little')

                entry_data[f"_{name}_int"] = val

                if fmt in ('hex', '0xhex'):
                    hex_digits = max(1, (bits + 3) // 4)
                    entry_data[name] = f"0x{val:0{hex_digits}x}"
                else:
                    entry_data[name] = val

            parsed_entries.append(entry_data)

        df = pd.DataFrame(parsed_entries)
        for col in df.columns:
            if col.startswith('_') and col.endswith('_int'):
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        return df

    @staticmethod
    def to_csv(df, path_or_buf=None, exclude_internal=True, **kwargs):
        """Export DataFrame to CSV, optionally excluding internal `_..._int` columns."""
        if exclude_internal:
            cols = [c for c in df.columns if not c.startswith('_')]
            return df[cols].to_csv(path_or_buf, index=False, **kwargs)
        return df.to_csv(path_or_buf, index=False, **kwargs)

    @staticmethod
    def format_text_table(df, exclude_internal=True):
        """Alias for the module-level format_text_table (for discoverability)."""
        return format_text_table(df, exclude_internal=exclude_internal)


# ==============================================================================
# 3. RAM Hex / Decimal Layout Dumper
# ==============================================================================
class MemoryDumper:
    """
    Renders RAM content formatted by bits per group and groups per row.
    Supports optional address prefix and hex/decimal output formats.
    """
    @staticmethod
    def dump(memory_map, start_addr, length_bytes, bits_per_group=32, groups_per_row=2, show_addr=True, mode='hex', endian='little'):
        """
        - `mode`: 'hex' or 'dec'
        - `endian`: 'little' or 'big'
        Returns string output suitable for display or logging.
        """
        data = memory_map.read_memory(start_addr, length_bytes)
        bytes_per_group = bits_per_group // 8
        if bytes_per_group <= 0:
            raise ValueError("bits_per_group must be at least 8.")

        bytes_per_row = bytes_per_group * groups_per_row

        lines = []
        for i in range(0, len(data), bytes_per_row):
            row_data = data[i : i + bytes_per_row]
            group_strings = []

            for j in range(0, len(row_data), bytes_per_group):
                chunk = row_data[j : j + bytes_per_group]
                if len(chunk) < bytes_per_group:
                    chunk = chunk.ljust(bytes_per_group, b'\x00')

                val = int.from_bytes(chunk, byteorder=endian)

                if mode == 'hex':
                    hex_digits = bytes_per_group * 2
                    group_strings.append(f"0x{val:0{hex_digits}x}")
                else:
                    group_strings.append(str(val))

            row_str = " ".join(group_strings)
            if show_addr:
                line = f"0x{start_addr + i:08x}: {row_str}"
            else:
                line = row_str
            lines.append(line)

        return "\n".join(lines)


# ==============================================================================
# 4. Register Specification Parsing (INI / CSV layout)
# ==============================================================================
class RegisterParser:
    """
    Parses a RAM region according to register specification specs defined in INI or CSV format.
    Fields specify DW offset (0x10 -> byte offset 0x10 * 4 = 0x40), field name, bit width, and optional format.
    """
    def __init__(self, memory_map):
        self.memory_map = memory_map

    @staticmethod
    def load_spec_from_csv(csv_path):
        """
        CSV format columns expected: offset_dw, name, bits, format (optional)
        """
        df = pd.read_csv(csv_path)
        # normalize column names
        df.columns = [c.strip().lower() for c in df.columns]
        if 'offset_dw' not in df.columns or 'name' not in df.columns or 'bits' not in df.columns:
            raise ValueError("CSV spec must contain 'offset_dw', 'name', and 'bits' columns.")
        
        # parse offset_dw and bits if hex strings
        df['offset_dw'] = df['offset_dw'].apply(lambda x: int(str(x), 0))
        df['bits'] = df['bits'].apply(lambda x: int(str(x), 0))
        if 'format' not in df.columns:
            df['format'] = 'hex'
        return df

    @staticmethod
    def load_spec_from_ini(ini_path):
        """
        INI format expected:
        [FIELD_NAME]
        offset_dw = 0x10
        bits = 8
        format = hex
        """
        config = configparser.ConfigParser()
        config.read(ini_path)
        
        rows = []
        for section in config.sections():
            dw = int(config.get(section, 'offset_dw', fallback='0'), 0)
            bits = int(config.get(section, 'bits', fallback='32'), 0)
            fmt = config.get(section, 'format', fallback='hex')
            rows.append({
                'name': section,
                'offset_dw': dw,
                'bits': bits,
                'format': fmt
            })
        return pd.DataFrame(rows)

    def parse_registers(self, start_addr, spec):
        """
        `spec`: DataFrame loaded from CSV or INI.
        Reads DW offsets (offset_dw * 4 bytes from start_addr) and bit slices.
        """
        if isinstance(spec, str):
            if spec.endswith('.csv'):
                spec_df = self.load_spec_from_csv(spec)
            elif spec.endswith('.ini'):
                spec_df = self.load_spec_from_ini(spec)
            else:
                raise ValueError("Spec file must be .csv or .ini")
        elif isinstance(spec, list):
            spec_df = pd.DataFrame(spec)
        else:
            spec_df = spec.copy()

        parsed_rows = []
        for _, row in spec_df.iterrows():
            dw_offset = int(row['offset_dw'])
            bit_offset_dw = int(row.get('bit_offset', 0))
            bits = int(row['bits'])
            name = row['name']
            fmt = str(row.get('format', 'hex')).lower()

            byte_addr = start_addr + (dw_offset * 4) + (bit_offset_dw // 8)
            bit_in_byte = bit_offset_dw % 8

            # Calculate bytes needed to read this bitfield
            total_bits_needed = bit_in_byte + bits
            bytes_needed = (total_bits_needed + 7) // 8

            raw_bytes = self.memory_map.read_memory(byte_addr, bytes_needed)
            full_val = int.from_bytes(raw_bytes, byteorder='little')
            val = (full_val >> bit_in_byte) & ((1 << bits) - 1)

            row_data = {
                'field_name': name,
                'dw_offset': f"0x{dw_offset:02x}",
                'byte_addr': f"0x{byte_addr:08x}",
                'bits': bits,
                'value_int': val
            }

            if fmt in ('hex', '0xhex'):
                hex_digits = max(1, (bits + 3) // 4)
                row_data['value'] = f"0x{val:0{hex_digits}x}"
            else:
                row_data['value'] = val

            row_data[f"_{name}_int"] = val
            parsed_rows.append(row_data)

        df = pd.DataFrame(parsed_rows)
        for col in df.columns:
            if col.startswith('_') and col.endswith('_int'):
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        return df


# ==============================================================================
# 5. Query Engine & Calculated Columns
# ==============================================================================
class QueryEngine:
    """
    Provides table querying, calculated column generation, and cross-table indexing.
    Querying compares pure integer values (not string representations).
    """
    @staticmethod
    def add_calculated_column(df, new_col_name, expr, format_as=None):
        """
        Adds a calculated column to `df`.
        `expr`: expression string evaluating existing integer columns or values, e.g.:
                `((Status >> 2) & 0xFF)` or `((_Status_int >> 2) & 0xFF)`
        `format_as`: optional 'hex' or 'dec' formatting for display.
        """
        working_df = df.copy()
        
        # numpy is imported at module level
        
        # Prepare evaluation environment with both plain names and _name_int columns as numpy arrays
        env = {'np': np}
        for col in working_df.columns:
            s = working_df[col]
            if col.startswith('_') and col.endswith('_int'):
                arr = pd.to_numeric(s, errors='coerce').fillna(0).to_numpy(dtype='int64')
                plain_name = col[1:-4]
                env[plain_name] = arr
                env[col] = arr
            else:
                try:
                    arr = pd.to_numeric(s, errors='raise').to_numpy(dtype='int64')
                    env[col] = arr
                except Exception:
                    env[col] = s.to_numpy()
        
        calculated_arr = eval(expr, {"__builtins__": None}, env)
        working_df[f"_{new_col_name}_int"] = pd.Series(calculated_arr, index=working_df.index).astype('int64')
        
        if format_as == 'hex':
            working_df[new_col_name] = working_df[f"_{new_col_name}_int"].apply(lambda x: f"0x{int(x):x}")
        else:
            working_df[new_col_name] = working_df[f"_{new_col_name}_int"]

        return working_df

    # Python/pandas keywords that should never be substituted in query expressions
    _QUERY_RESERVED = frozenset({
        'and', 'or', 'not', 'in', 'is', 'if', 'else', 'elif',
        'for', 'while', 'True', 'False', 'None', 'del', 'from',
        'import', 'as', 'with', 'try', 'except', 'finally',
        'raise', 'return', 'class', 'def', 'pass', 'break',
        'continue', 'lambda', 'yield', 'global', 'nonlocal',
        'assert',
    })

    @staticmethod
    def query_table(df, condition_str):
        """
        Executes query condition on DataFrame.
        Automatically converts field references to integer columns if `_field_int` exists.
        Terms are processed longest-first to prevent partial substitution of column names.
        Python keywords are excluded from substitution.
        Example: `query_table(df, "Status > 10 and c == 256")`
        """
        expr_terms = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', condition_str)
        # Deduplicate and sort by length descending to substitute longest matches first
        unique_terms = sorted(set(expr_terms), key=len, reverse=True)
        transformed_condition = condition_str

        for term in unique_terms:
            if term in QueryEngine._QUERY_RESERVED:
                continue
            if f"_{term}_int" in df.columns:
                transformed_condition = re.sub(rf'\b{term}\b', f"_{term}_int", transformed_condition)

        return df.query(transformed_condition)

    @staticmethod
    def use_field_as_index(source_df, source_field_name, target_df):
        """
        Retrieves the integer value of `source_field_name` from row 0 of `source_df`,
        and uses it as an entry index to retrieve the corresponding row in `target_df`.
        Warns if `source_df` contains more than one row.
        """
        if len(source_df) > 1:
            warnings.warn(
                f"use_field_as_index: source_df has {len(source_df)} rows; only row 0 will be used.",
                UserWarning,
                stacklevel=2,
            )
        int_col = f"_{source_field_name}_int" if f"_{source_field_name}_int" in source_df.columns else source_field_name
        index_val = int(source_df.iloc[0][int_col])

        if index_val >= len(target_df):
            raise IndexError(f"Index value {index_val} from {source_field_name} out of bounds for target table (size {len(target_df)})")
        
        return target_df.iloc[[index_val]]


# ==============================================================================
# Helper Utility: Formatted Text Table Alignment
# ==============================================================================
def format_text_table(df, exclude_internal=True):
    """Formats DataFrame as aligned plain-text table, hiding internal `_..._int` columns by default."""
    if exclude_internal:
        cols = [c for c in df.columns if not c.startswith('_')]
        return df[cols].to_string(index=False)
    return df.to_string(index=False)
