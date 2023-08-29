use goblin::elf::*;

use std::fs::File;
use std::io::Read;

use std::ops::*;

pub struct Binary {
    program_header: Vec<ProgramHeader>,
    bytes: Vec<u8>,
}

impl Binary {
    // from path of the exe file to Binary instance
    pub fn from_elf(path: String) -> Self {
        // INITIALIZATION: file read, length
        let mut file = File::open(path).map_err(|_| "open file error").unwrap();
        let file_len = file
            .metadata()
            .map_err(|_| "get metadata error")
            .unwrap()
            .len();

        // INITIALIZATION: vector of bytes
        let mut contents = vec![0; file_len as usize];
        file.read_exact(&mut contents[..])
            .map_err(|_| "read header error")
            .unwrap();

        // INITIALIZATION: elf file
        let elf = Elf::parse(&contents[..])
            .map_err(|_| "cannot parse elf file error")
            .unwrap();

        Binary {
            program_header: elf.program_headers,
            bytes: contents,
        }
    }

    // slice of bytes at a given virtual address range or error:invalid
    pub fn virtual_address_range<T: RangeBounds<u64>>(&self, range: T) -> Result<&[u8], String> {
        // start bound
        let start: u64 = match range.start_bound() {
            Bound::Unbounded => 0,
            Bound::Excluded(num) => *num + 1,
            Bound::Included(num) => *num,
        };

        // index of program containing given virtual address range
        let segment = &self
            .program_header
            .iter()
            .position(
                |x|
                    // p_type = "PT_LOAD"
                    x.p_type == 1 &&
                    // given va range is inside the range of program
                    x.p_vaddr <= start &&
                    start <= x.p_vaddr + x.p_filesz, // range.end <= x.p_vaddr + x.p_filesz
            )
            .ok_or(String::from("invalid virtual address range error"))?;

        let segment = &self.program_header[*segment];

        // end bound
        let end: u64 = match range.end_bound() {
            Bound::Unbounded => segment.p_vaddr + segment.p_filesz,
            Bound::Excluded(num) => *num - 1,
            Bound::Included(num) => *num,
        };

        if end > segment.p_vaddr + segment.p_filesz {
            Err(String::from("invalid virtual address range error"))
        } else {
            Ok(&self.bytes[
                // convert the virtual address to file address
                (start - segment.p_vaddr + segment.p_offset) as usize .. (end - segment.p_vaddr + segment.p_offset) as usize
            ])
        }
    }
}
