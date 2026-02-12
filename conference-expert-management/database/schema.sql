-- 会务专家信息管理系统数据库脚本
-- 数据库: conference_expert_management

-- 创建数据库
CREATE DATABASE IF NOT EXISTS conference_expert_management 
    DEFAULT CHARACTER SET utf8mb4 
    DEFAULT COLLATE utf8mb4_unicode_ci;

USE conference_expert_management;

-- 专家信息表
CREATE TABLE IF NOT EXISTS experts (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '专家ID',
    name VARCHAR(100) NOT NULL COMMENT '姓名',
    email VARCHAR(255) UNIQUE NOT NULL COMMENT '邮箱',
    phone VARCHAR(20) COMMENT '电话',
    professional_field VARCHAR(255) COMMENT '专业领域',
    title VARCHAR(100) COMMENT '职称',
    organization VARCHAR(255) COMMENT '所属机构',
    address VARCHAR(500) COMMENT '地址',
    bio TEXT COMMENT '简介',
    contact_status VARCHAR(50) DEFAULT '待联系' COMMENT '联系状态：待联系/已联系/确认参加/暂不参与',
    is_deleted TINYINT(1) DEFAULT 0 COMMENT '是否删除（软删除）',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_name (name),
    INDEX idx_field (professional_field),
    INDEX idx_status (contact_status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='专家信息表';

-- 联系记录表
CREATE TABLE IF NOT EXISTS contact_records (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '记录ID',
    expert_id BIGINT NOT NULL COMMENT '专家ID',
    contact_time DATETIME NOT NULL COMMENT '联系时间',
    contact_method VARCHAR(50) COMMENT '联系方式：电话/邮件/微信/面谈',
    content TEXT COMMENT '联系内容',
    operator VARCHAR(100) COMMENT '操作人',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    FOREIGN KEY (expert_id) REFERENCES experts(id) ON DELETE CASCADE,
    INDEX idx_expert_time (expert_id, contact_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='联系记录表';

-- 会务参与记录表
CREATE TABLE IF NOT EXISTS conference_records (
    id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '记录ID',
    expert_id BIGINT NOT NULL COMMENT '专家ID',
    conference_name VARCHAR(255) NOT NULL COMMENT '会议名称',
    conference_date DATE COMMENT '会议日期',
    role VARCHAR(100) COMMENT '参与角色：演讲嘉宾/主持人/参会者',
    topic VARCHAR(500) COMMENT '演讲主题',
    feedback TEXT COMMENT '专家反馈',
    is_completed TINYINT(1) DEFAULT 0 COMMENT '是否完成反馈',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    FOREIGN KEY (expert_id) REFERENCES experts(id) ON DELETE CASCADE,
    INDEX idx_expert_date (expert_id, conference_date),
    UNIQUE KEY uk_expert_conference (expert_id, conference_name, conference_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='会务参与记录表';
