package com.example.expert.service;

import com.example.expert.dto.ExpertDTO;
import com.example.expert.entity.Expert;
import com.example.expert.repository.ExpertRepository;
import lombok.RequiredArgsConstructor;
import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

@Service
@RequiredArgsConstructor
public class ExcelService {
    
    private final ExpertRepository expertRepository;
    
    public ByteArrayInputStream exportExperts(List<ExpertDTO> experts) throws IOException {
        try (Workbook workbook = new XSSFWorkbook(); ByteArrayOutputStream out = new ByteArrayOutputStream()) {
            Sheet sheet = workbook.createSheet("Experts");
            
            // Header
            Row headerRow = sheet.createRow(0);
            String[] headers = {"姓名", "邮箱", "电话", "专业领域", "职称", "所属机构", "地址", "联系状态"};
            for (int i = 0; i < headers.length; i++) {
                Cell cell = headerRow.createCell(i);
                cell.setCellValue(headers[i]);
            }
            
            // Data
            int rowIdx = 1;
            for (ExpertDTO expert : experts) {
                Row row = sheet.createRow(rowIdx++);
                row.createCell(0).setCellValue(expert.getName());
                row.createCell(1).setCellValue(expert.getEmail());
                row.createCell(2).setCellValue(expert.getPhone());
                row.createCell(3).setCellValue(expert.getProfessionalField());
                row.createCell(4).setCellValue(expert.getTitle());
                row.createCell(5).setCellValue(expert.getOrganization());
                row.createCell(6).setCellValue(expert.getAddress());
                row.createCell(7).setCellValue(expert.getContactStatus());
            }
            
            workbook.write(out);
            return new ByteArrayInputStream(out.toByteArray());
        }
    }
    
    public ByteArrayInputStream getImportTemplate() throws IOException {
        try (Workbook workbook = new XSSFWorkbook(); ByteArrayOutputStream out = new ByteArrayOutputStream()) {
            Sheet sheet = workbook.createSheet("Template");
            
            // Header
            Row headerRow = sheet.createRow(0);
            String[] headers = {"姓名*", "邮箱*", "电话", "专业领域", "职称", "所属机构", "地址"};
            for (int i = 0; i < headers.length; i++) {
                Cell cell = headerRow.createCell(i);
                cell.setCellValue(headers[i]);
            }
            
            // Sample data
            Row sampleRow = sheet.createRow(1);
            sampleRow.createCell(0).setCellValue("张三");
            sampleRow.createCell(1).setCellValue("zhangsan@example.com");
            sampleRow.createCell(2).setCellValue("13800138000");
            sampleRow.createCell(3).setCellValue("人工智能");
            sampleRow.createCell(4).setCellValue("教授");
            sampleRow.createCell(5).setCellValue("清华大学");
            sampleRow.createCell(6).setCellValue("北京市");
            
            workbook.write(out);
            return new ByteArrayInputStream(out.toByteArray());
        }
    }
    
    public List<ExpertDTO> importExperts(MultipartFile file) throws IOException {
        List<ExpertDTO> importedExperts = new ArrayList<>();
        List<String> errors = new ArrayList<>();
        
        try (Workbook workbook = new XSSFWorkbook(file.getInputStream())) {
            Sheet sheet = workbook.getSheetAt(0);
            
            for (int i = 1; i <= sheet.getLastRowNum(); i++) {
                Row row = sheet.getRow(i);
                if (row == null) continue;
                
                try {
                    ExpertDTO dto = new ExpertDTO();
                    dto.setName(getCellValue(row.getCell(0)));
                    dto.setEmail(getCellValue(row.getCell(1)));
                    dto.setPhone(getCellValue(row.getCell(2)));
                    dto.setProfessionalField(getCellValue(row.getCell(3)));
                    dto.setTitle(getCellValue(row.getCell(4)));
                    dto.setOrganization(getCellValue(row.getCell(5)));
                    dto.setAddress(getCellValue(row.getCell(6)));
                    dto.setContactStatus("待联系");
                    
                    // Validation
                    if (dto.getName() == null || dto.getName().isEmpty()) {
                        errors.add("Row " + (i + 1) + ": 姓名不能为空");
                        continue;
                    }
                    if (dto.getEmail() == null || dto.getEmail().isEmpty()) {
                        errors.add("Row " + (i + 1) + ": 邮箱不能为空");
                        continue;
                    }
                    
                    // Check duplicate email
                    if (expertRepository.existsByEmailAndIsDeletedFalse(dto.getEmail())) {
                        errors.add("Row " + (i + 1) + ": 邮箱 " + dto.getEmail() + " 已存在，已跳过");
                        continue;
                    }
                    
                    importedExperts.add(dto);
                } catch (Exception e) {
                    errors.add("Row " + (i + 1) + ": " + e.getMessage());
                }
            }
        }
        
        if (!errors.isEmpty()) {
            throw new RuntimeException("导入完成，但有以下错误：\n" + String.join("\n", errors));
        }
        
        return importedExperts;
    }
    
    private String getCellValue(Cell cell) {
        if (cell == null) return null;
        switch (cell.getCellType()) {
            case STRING:
                return cell.getStringCellValue();
            case NUMERIC:
                return String.valueOf((long) cell.getNumericCellValue());
            default:
                return null;
        }
    }
}
